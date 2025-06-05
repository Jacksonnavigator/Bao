import streamlit as st
from bao_game import BaoGame
from minimax_ai import MinimaxAI

# Initialize game state
if 'game' not in st.session_state:
    st.session_state.game = BaoGame()
    st.session_state.ai = MinimaxAI(max_depth=2)
    st.session_state.status = ""
    st.session_state.move_history = []
    st.session_state.last_move = None

game = st.session_state.game
ai = st.session_state.ai

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
