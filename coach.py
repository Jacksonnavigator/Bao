import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from neural_net import BaoNet, board_to_tensor, save_model, load_model
from replay_buffer import ReplayBuffer
from self_play import self_play_game
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
