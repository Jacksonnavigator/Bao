import torch
import numpy as np
from collections import deque
from mcts import MCTS
from neural_net import board_to_tensor


def self_play_game(game_factory, net, n_simulations=200, device=None, temp_threshold=8):
    """
    Play one self-play game using MCTS guided by `net`.
    - temp_threshold: number of initial moves where temperature=1 (exploration). After that temperature->0 (greedy).
    Returns list of (state_tensor, policy_flat, value) examples where policy_flat is a 1D numpy array of length rows*cols (probabilities).
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    mcts = MCTS(game_factory, net, n_simulations=n_simulations, device=device)
    game = game_factory()
    examples = []
    move_count = 0

    while not game.is_game_over():
        policy, root = mcts.run(game)
        # policy is visit-count distribution normalized (rows x cols)

        # record state and policy target (flattened)
        state_tensor = board_to_tensor(game.board, game.current_player)
        policy_flat = policy.flatten().astype(np.float32)
        # ensure numerical stability
        if policy_flat.sum() <= 0:
            # fallback to uniform over legal moves
            legal = game.get_legal_moves()
            policy_flat = np.zeros(game.rows * game.cols, dtype=np.float32)
            for (r, c) in legal:
                policy_flat[r * game.cols + c] = 1.0
            policy_flat /= policy_flat.sum()

        examples.append((state_tensor, policy_flat, None))

        # temperature schedule: exploratory for first temp_threshold moves
        temp = 1.0 if move_count < temp_threshold else 0.0
        flat = policy_flat.copy()
        if temp == 0.0:
            move_idx = int(np.argmax(flat))
        else:
            # apply temperature by raising probabilities to 1/temp
            # with temp=1 this is unchanged; lower temp sharpens distribution
            probs = flat / (flat.sum() + 1e-12)
            # sample according to probs
            move_idx = np.random.choice(len(probs), p=probs)

        r = move_idx // game.cols
        c = move_idx % game.cols
        game.make_move(r, c)
        move_count += 1

    # determine game outcome: for current player who has no moves, the opponent is the winner
    winner = 3 - game.current_player
    # assign values relative to the player to move in each stored state
    filled = []
    for state, policy, _ in examples:
        # state[1] channel contains player mask
        player_ch = state[1].numpy()
        if player_ch[0].sum() > 0:
            ply = 1
        else:
            ply = 2
        val = 1.0 if winner == ply else -1.0
        filled.append((state, policy, val))

    return filled
