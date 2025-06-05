class BaoGame:
    def __init__(self):
        self.rows = 4
        self.cols = 8
        self.board = [[4 for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = 1
        self.player_rows = {1: [0, 1], 2: [2, 3]}

    def get_legal_moves(self):
        moves = []
        for r in self.player_rows[self.current_player]:
            for c in range(self.cols):
                if self.board[r][c] > 0:
                    moves.append((r, c))
        return moves

    def make_move(self, row, col):
        seeds = self.board[row][col]
        if seeds == 0:
            return False  # invalid

        self.board[row][col] = 0
        r, c = row, col

        while seeds > 0:
            c += 1
            if c == self.cols:
                c = 0
                r = (r + 1) % self.rows

            if (r, c) == (row, col):
                continue  # skip starting pit

            self.board[r][c] += 1
            seeds -= 1

        # Capture: simplified
        opponent_rows = self.player_rows[3 - self.current_player]
        if r in opponent_rows and self.board[r][c] in [2, 3]:
            captured = self.board[r][c]
            self.board[r][c] = 0
            # Optional: add captured to player

        self.current_player = 3 - self.current_player
        return True

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def get_board_state(self):
        return self.board, self.current_player
