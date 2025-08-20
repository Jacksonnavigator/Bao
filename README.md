# Bao - AlphaZero-style implementation

This repository contains an AlphaZero-style training pipeline for the Bao board game.

Quickstart (PowerShell):

1. Create and activate Python environment

   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies (adjust torch CUDA wheel to your CUDA version)

   pip install --upgrade pip; pip install -r requirements.txt

   # For CUDA example (replace cu118 with your CUDA version):

   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

3. Run a quick smoke test

   python -c "from train import main; main(iters=1,self_play_games=1,n_simulations=50,train_epochs=1,batch_size=8,lr=1e-3,eval_games=2)"

4. Launch Streamlit UI

   streamlit run app.py

5. Monitor TensorBoard

   tensorboard --logdir runs

Notes:

- Increase `n_simulations` and `self_play_games` for stronger training.
- If you have multiple GPUs, consider modifying self-play to run parallel processes pinned to separate devices.
- This is a starting prototype — further improvements include parallel self-play, improved network architecture, data augmentation, and hyperparameter tuning.
