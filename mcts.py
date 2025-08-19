import math
import time
import numpy as np
import torch
from collections import defaultdict

class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, game_factory, net, cpuct=1.0, n_simulations=800, device=None):
        self.game_factory = game_factory
        self.net = net
        self.cpuct = cpuct
        self.n_simulations = n_simulations
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def run(self, game):
        root = MCTSNode()
        # use network to get priors and value for root
        board, player = game.get_board_state()
        state_tensor = self._make_tensor(board, player).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(state_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            value = float(value.cpu().numpy()[0])

        legal = game.get_legal_moves()
        total_actions = game.rows * game.cols
        priors = np.zeros(total_actions, dtype=np.float32)
        for (r, c) in legal:
            priors[r * game.cols + c] = probs[r * game.cols + c]

        for (r, c) in legal:
            idx = r * game.cols + c
            root.children[idx] = MCTSNode(prior=priors[idx])

        for _ in range(self.n_simulations):
            node = root
            sim_game = self.game_factory()  # get fresh game object
            sim_game.board = [row[:] for row in board]
            sim_game.current_player = player

            # select
            search_path = [node]
            while node.children:
                max_ucb = -float('inf')
                best_idx = None
                best_child = None
                for a, child in node.children.items():
                    ucb = child.value() + self.cpuct * child.prior * math.sqrt(node.visit_count + 1) / (1 + child.visit_count)
                    if ucb > max_ucb:
                        max_ucb = ucb
                        best_idx = a
                        best_child = child
                # apply move
                r = best_idx // sim_game.cols
                c = best_idx % sim_game.cols
                sim_game.make_move(r, c)
                node = best_child
                search_path.append(node)

            # expand
            b, p = sim_game.get_board_state()
            t = self._make_tensor(b, p).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.net(t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                value = float(value.cpu().numpy()[0])

            legal = sim_game.get_legal_moves()
            for (r, c) in legal:
                idx = r * sim_game.cols + c
                if idx not in node.children:
                    node.children[idx] = MCTSNode(prior=probs[idx])

            # backpropagate
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value
                value = -value  # switch perspective

        # compute visit probabilities
        visit_counts = np.zeros(game.rows * game.cols, dtype=np.float32)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count
        policy = visit_counts / (np.sum(visit_counts) + 1e-8)

        return policy.reshape(game.rows, game.cols), root

    def _make_tensor(self, board, player):
        import numpy as np
        arr = np.array(board, dtype=np.float32)
        norm = arr / 12.0
        ch0 = norm
        ch1 = np.zeros_like(arr, dtype=np.float32)
        if player == 1:
            ch1[0:2, :] = 1.0
        else:
            ch1[2:4, :] = 1.0
        tensor = np.stack([ch0, ch1], axis=0)
        return torch.from_numpy(tensor)
