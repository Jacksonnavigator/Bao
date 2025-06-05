import random
import copy

class MinimaxAI:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def choose_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None
        best_score = float('-inf')
        best_move = random.choice(moves)

        for move in moves:
            simulated_game = copy.deepcopy(game)
            simulated_game.make_move(*move)
            score = self.minimax(simulated_game, self.max_depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, game, depth, is_maximizing):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game)

        moves = game.get_legal_moves()
        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                sim = copy.deepcopy(game)
                sim.make_move(*move)
                max_eval = max(max_eval, self.minimax(sim, depth - 1, False))
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                sim = copy.deepcopy(game)
                sim.make_move(*move)
                min_eval = min(min_eval, self.minimax(sim, depth - 1, True))
            return min_eval

    def evaluate(self, game):
        board, player = game.get_board_state()
        p1 = sum(board[r][c] for r in game.player_rows[1] for c in range(game.cols))
        p2 = sum(board[r][c] for r in game.player_rows[2] for c in range(game.cols))
        return p1 - p2 if player == 1 else p2 - p1
