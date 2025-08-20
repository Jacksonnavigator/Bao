import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from neural_net import BaoNet, board_to_tensor, save_model, load_model
from replay_buffer import ReplayBuffer
from self_play import self_play_game
from mcts import MCTS
import numpy as np
import os

class Coach:
    def __init__(self, game_factory, device=None, log_dir='runs'):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.net = BaoNet().to(self.device)
        self.replay = ReplayBuffer(capacity=20000)
        self.game_factory = game_factory
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_step = 0
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def self_play_and_fill(self, n_games=10, n_simulations=200):
        for _ in range(n_games):
            examples = self_play_game(self.game_factory, self.net, n_simulations=n_simulations, device=self.device)
            self.replay.push(examples)

    def train(self, epochs=5, batch_size=64, lr=1e-3, epochs_per_checkpoint=1):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(epochs):
            if len(self.replay) < batch_size:
                print('Not enough samples in replay to train yet')
                return

            # compute number of batches available
            n_batches = max(1, len(self.replay) // batch_size)
            for _ in range(n_batches):
                states, policies, values = self.replay.sample(batch_size)
                states = states.to(self.device)
                policies = policies.view(policies.size(0), -1).to(self.device)
                values = values.to(self.device)

                logits, preds = self.net(states)
                # policy loss: use KL or cross-entropy with soft targets
                log_probs = torch.log_softmax(logits, dim=1)
                loss_p = - (policies * log_probs).sum(dim=1).mean()
                # value loss
                loss_v = torch.nn.functional.mse_loss(preds, values)
                loss = loss_p + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                self.writer.add_scalar('loss/total', loss.item(), self.train_step)
                self.writer.add_scalar('loss/policy', loss_p.item(), self.train_step)
                self.writer.add_scalar('loss/value', loss_v.item(), self.train_step)
                self.writer.add_scalar('replay/size', len(self.replay), self.train_step)
                self.train_step += 1

            # checkpoint
            if (epoch + 1) % epochs_per_checkpoint == 0:
                path = os.path.join(self.checkpoint_dir, f'model_step_{self.train_step}.pth')
                save_model(self.net, path)
                print(f'Saved checkpoint to {path}')

        # final save
        save_model(self.net, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        print('Training finished and model saved.')

    def play_game_between(self, net1, net2, n_simulations=100):
        """Play one game between net1 (player 1) and net2 (player 2) using MCTS-guided play.
        Returns winner (1 or 2).
        """
        game = self.game_factory()
        while not game.is_game_over():
            current = game.current_player
            # choose the network for the current player
            net = net1 if current == 1 else net2
            mcts = MCTS(self.game_factory, net, n_simulations=n_simulations, device=self.device)
            policy, root = mcts.run(game)
            flat = policy.flatten()
            move_idx = int(np.argmax(flat))
            r = move_idx // game.cols
            c = move_idx % game.cols
            game.make_move(r, c)
        winner = 3 - game.current_player
        return winner

    def evaluate_challenger(self, challenger_net, n_games=20, n_simulations=100, win_threshold=0.55):
        """Evaluate challenger_net against the current incumbent (if any).
        If challenger win rate > win_threshold, promote and save as best model.
        Returns True if promoted.
        """
        incumbent_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        incumbent = None
        if os.path.exists(incumbent_path):
            incumbent = BaoNet().to(self.device)
            load_model(incumbent, incumbent_path, device=self.device)

        # If no incumbent, promote challenger immediately
        if incumbent is None:
            save_model(challenger_net, incumbent_path)
            print('No incumbent found — promoted challenger to best_model.pth')
            return True

        challenger_wins = 0
        incumbent_wins = 0

        for i in range(n_games):
            # alternate which network plays as player 1
            if i % 2 == 0:
                p1, p2 = challenger_net, incumbent
                winner = self.play_game_between(p1, p2, n_simulations=n_simulations)
                if winner == 1:
                    challenger_wins += 1
                else:
                    incumbent_wins += 1
            else:
                p1, p2 = incumbent, challenger_net
                winner = self.play_game_between(p1, p2, n_simulations=n_simulations)
                # if winner==2 challenger was player2
                if winner == 2:
                    challenger_wins += 1
                else:
                    incumbent_wins += 1

        win_rate = challenger_wins / float(n_games)
        self.writer.add_scalar('evaluation/challenger_win_rate', win_rate, self.train_step)
        print(f'Challenger wins: {challenger_wins}, Incumbent wins: {incumbent_wins}, win_rate={win_rate:.3f}')

        if win_rate > win_threshold:
            save_model(challenger_net, incumbent_path)
            print('Challenger promoted to incumbent.')
            return True
        else:
            print('Challenger not promoted.')
            return False
