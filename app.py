import streamlit as st
from bao_game import BaoGame
from minimax_ai import MinimaxAI
import torch
from neural_net import BaoNet, board_to_tensor, load_model
from mcts import MCTS
from coach import Coach

# Initialize game state
if 'game' not in st.session_state:
    st.session_state.game = BaoGame()
    st.session_state.ai = MinimaxAI(max_depth=2)
    st.session_state.status = ""
    st.session_state.move_history = []
    st.session_state.last_move = None
    st.session_state.coach = Coach(lambda: BaoGame(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

game = st.session_state.game
ai = st.session_state.ai
coach = st.session_state.coach

st.set_page_config(page_title="Bao Game AI", layout="wide")
st.title("🕹️ Bao Game - Play vs Strategic AI")

board, current_player = game.get_board_state()

# Scores
def get_scores():
    p1 = sum(board[r][c] for r in game.player_rows[1] for c in range(game.cols))
    p2 = sum(board[r][c] for r in game.player_rows[2] for c in range(game.cols))
    return p1, p2

p1_score, p2_score = get_scores()
st.markdown(f"### 🔵 Player 1 (You): {p1_score} &nbsp;&nbsp;&nbsp; 🔴 Player 2 (AI): {p2_score}")

# Game status
if game.is_game_over():
    winner = "You 🎉" if current_player == 2 else "AI 🤖"
    st.success(f"🎯 Game Over! **{winner} wins!**")
    st.balloons()
else:
    st.markdown(f"#### 🎲 Turn: {'You (🔵)' if current_player == 1 else 'AI (🔴)'}")

# Board
def display_board():
    for r_idx, row in enumerate(board):
        cols = st.columns(len(row))
        for c_idx, seeds in enumerate(row):
            label = f"{seeds}"
            if st.session_state.last_move == (r_idx, c_idx):
                label = f"🎯 {seeds}"

            if current_player == 1 and (r_idx, c_idx) in game.get_legal_moves():
                if cols[c_idx].button(label, key=f"{r_idx}_{c_idx}"):
                    game.make_move(r_idx, c_idx)
                    st.session_state.last_move = (r_idx, c_idx)
                    st.session_state.move_history.append((1, (r_idx, c_idx)))
                    st.experimental_rerun()
            else:
                cols[c_idx].markdown(f"### {label}")

display_board()

# AI Move
if current_player == 2 and not game.is_game_over():
    ai_move = ai.choose_move(game)
    if ai_move:
        game.make_move(*ai_move)
        st.session_state.last_move = ai_move
        st.session_state.move_history.append((2, ai_move))
        st.experimental_rerun()

# Move history sidebar
with st.sidebar:
    st.header("📜 Move History")
    for i, (player, move) in enumerate(st.session_state.move_history, 1):
        icon = "🔵" if player == 1 else "🔴"
        st.markdown(f"{i}. {icon} {move}")
    if st.button("🔄 Restart Game"):
        st.session_state.game = BaoGame()
        st.session_state.move_history = []
        st.session_state.last_move = None
        st.experimental_rerun()

# Training controls
st.sidebar.header("⚙️ Training")
if st.sidebar.button("▶️ Run one self-play game"):
    with st.spinner("Running self-play..."):
        examples = coach.self_play_and_fill(n_games=1, n_simulations=50)
        st.sidebar.success("Self-play finished and added to replay buffer.")

if st.sidebar.button("🧠 Train one epoch"):
    with st.spinner("Training..."):
        coach.train(epochs=1, batch_size=32)
        st.sidebar.success("Training epoch completed.")

# MCTS visualization
st.sidebar.header("🔎 MCTS")
if st.sidebar.button("Run MCTS for current position"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = coach.net
    mcts = MCTS(lambda: BaoGame(), net, n_simulations=200, device=device)
    policy, root = mcts.run(game)
    import numpy as np
    heat = (policy * 100).astype(int)
    st.markdown("### MCTS visit distribution (policy)")
    for r in range(game.rows):
        cols = st.columns(game.cols)
        for c in range(game.cols):
            cols[c].markdown(f"**{heat[r,c]}**")
