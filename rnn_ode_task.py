import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import data
from RNN_ODE_functions import train_non_adap_models, fit_non_adap_models_grids


class WindowDataset(Dataset):
    def __init__(self, windows: torch.Tensor, ts: torch.Tensor):
        self.windows = windows
        self.ts = ts

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx):
        return self.windows[idx], self.ts[idx]


def build_time_grid(num_samples: int, num_steps: int, t_end: float = 400.0) -> torch.Tensor:
    base_ts = torch.linspace(0.0, t_end, num_steps + 1, dtype=torch.float32)[:-1]
    return base_ts.unsqueeze(0).repeat(num_samples, 1)


def main():
    parser = argparse.ArgumentParser(description='Train RNN-ODE using helper utilities (LEM comparison ground).')
    parser.add_argument('--nhid', type=int, default=128, help='latent/hidden size (matches LEM baseline)')
    parser.add_argument('--epochs', type=int, default=400, help='number of training iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='computing device')
    parser.add_argument('--batch', type=int, default=32, help='batch size (same as LEM setup)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate used in spiral_example')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    print(args)

    ninp = 1

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate FitzHugh–Nagumo data as in the LEM baseline
    train_x, _ = data.get_data(128)
    valid_x, _ = data.get_data(128)
    test_x, _ = data.get_data(1024)
    print('Finished generating data')

    train_windows = torch.from_numpy(train_x).float()
    valid_windows = torch.from_numpy(valid_x).float()
    test_windows = torch.from_numpy(test_x).float()

    num_steps = train_windows.shape[1]
    train_ts = build_time_grid(train_windows.shape[0], num_steps)
    valid_ts = build_time_grid(valid_windows.shape[0], num_steps)
    test_ts = build_time_grid(test_windows.shape[0], num_steps)

    buffer_start_steps = 2
    time_scale = 10.0
    method = 'naiveEuler'
    verbose = 1
    rescale_const = 1.0
    thres1 = 6.5e3
    thres2 = 6.5e3
    weight = (1, 0)
    num_grids = num_steps

    # Apply buffer timestamp shift (same trick as notebook helper)
    if buffer_start_steps > 0:
        deltat = train_ts[0, -1] - train_ts[0, -2]
        shift = torch.arange(-deltat * buffer_start_steps, 0, deltat, dtype=train_ts.dtype)
        train_ts = train_ts.clone()
        train_ts[:, :buffer_start_steps] = train_ts[:, :buffer_start_steps] + shift

    train_dataset = WindowDataset(train_windows, train_ts)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)
    input_steps = train_windows.shape[1]

    flag, odefunc, outputfunc = train_non_adap_models(
        train_loader,
        input_steps,
        valid_windows,
        valid_ts,
        num_grids,
        verbose,
        lr=args.lr,
        n_iter=args.epochs,
        method=method,
        thres1=thres1,
        thres2=thres2,
        weight=weight,
        obs_dim=ninp,
        n_hidden=args.nhid,
        n_latent=args.nhid,
        num_train_windows=train_windows.shape[0],
        time_scale=time_scale,
        buffer_start_steps=buffer_start_steps,
        rescale_const=rescale_const,
        device=args.device,
    )

    if not flag:
        raise SystemExit('Training failed: loss thresholds exceeded.')

    # Validation RMSE (fit metric) using helper routine
    _, valid_rmse_tensor, _ = fit_non_adap_models_grids(
        odefunc,
        outputfunc,
        valid_windows,
        valid_ts,
        num_grids,
        method=method,
        buffer_start_steps=buffer_start_steps,
        n_latent=args.nhid,
        obs_dim=ninp,
        rescale_const=rescale_const,
        time_scale=time_scale,
    )
    valid_rmse = valid_rmse_tensor.mean().item()

    # Test RMSE on large holdout set
    _, test_rmse_tensor, _ = fit_non_adap_models_grids(
        odefunc,
        outputfunc,
        test_windows,
        test_ts,
        num_grids,
        method=method,
        buffer_start_steps=buffer_start_steps,
        n_latent=args.nhid,
        obs_dim=ninp,
        rescale_const=rescale_const,
        time_scale=time_scale,
    )
    test_rmse = test_rmse_tensor.mean().item()

    Path('result').mkdir(parents=True, exist_ok=True)
    with open('result/FNH_RNN_ODE.txt', 'a') as f:
        f.write(f'valid rmse: {valid_rmse}\n')
        f.write(f'test rmse: {test_rmse}\n')

    print('Done training RNN-ODE via helper utilities.')
    print(f'Validation RMSE: {valid_rmse:.6f}, Test RMSE: {test_rmse:.6f}')


if __name__ == '__main__':
    main()
