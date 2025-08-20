import argparse
import torch
from coach import Coach
from bao_game import BaoGame


def main(iters=10, self_play_games=10, n_simulations=200, train_epochs=1, batch_size=64, lr=1e-3, eval_games=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coach = Coach(lambda: BaoGame(), device=device)

    for it in range(1, iters + 1):
        print(f'== Iteration {it} - self-play ({self_play_games} games) ==')
        coach.self_play_and_fill(n_games=self_play_games, n_simulations=n_simulations)

        print('== Training ==')
        coach.train(epochs=train_epochs, batch_size=batch_size, lr=lr)

        print('== Evaluation ==')
        promoted = coach.evaluate_challenger(coach.net, n_games=eval_games, n_simulations=n_simulations)
        print(f'Evaluation result: promoted={promoted}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--self_play_games', type=int, default=10)
    parser.add_argument('--n_simulations', type=int, default=200)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_games', type=int, default=20)
    args = parser.parse_args()
    main(args.iters, args.self_play_games, args.n_simulations, args.train_epochs, args.batch_size, args.lr, args.eval_games)
