from torch import nn, optim, Tensor
import torch
import network
import torch.nn.utils
import data
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=400,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00904,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--runs', type=int, default=25,
                    help='number of experiments to run')

args = parser.parse_args()
print(args)

device = torch.device(str(args.device))

ninp = 1
nout = 1

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

train_x, train_y = data.get_data(128)
valid_x, valid_y = data.get_data(128)
test_x, test_y = data.get_data(1024)
print('Finished generating data')

## Train data:
train_dataset = TensorDataset(Tensor(train_x).float(), Tensor(train_y))
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)

## Valid data
valid_dataset = TensorDataset(Tensor(valid_x).float(), Tensor(valid_y).float())
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=128)

## Test data
test_dataset = TensorDataset(Tensor(test_x).float(), Tensor(test_y).float())
testloader = DataLoader(test_dataset, shuffle=False, batch_size=128)


objective = nn.MSELoss()


def eval_model(model, dataloader):
    model.eval()
    loss = None
    with torch.no_grad():
        for x, y in dataloader:
            y = y.permute(1, 0, 2)
            x = x.permute(1, 0, 2)
            out = model(x.to(device))
            loss = torch.sqrt(objective(out, y.to(device))).item()
    return loss


def run_experiment(run_idx):
    run_seed = args.seed + run_idx
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    model = network.LEM(ninp, args.nhid, nout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    final_test_loss = None

    Path('result').mkdir(parents=True, exist_ok=True)
    result_path = Path('result/FNH_LEM.txt')

    with result_path.open('a') as f:
        f.write(f'run {run_idx + 1}\n')
        for epoch in range(args.epochs):
            model.train()
            for x, y in trainloader:
                y = y.permute(1, 0, 2)
                x = x.permute(1, 0, 2)
                optimizer.zero_grad()
                out = model(x.to(device))
                loss = objective(out, y.to(device))
                loss.backward()
                optimizer.step()

            valid_loss = eval_model(model, validloader)
            test_loss = eval_model(model, testloader)

            if final_test_loss is None:
                final_test_loss = test_loss

            if valid_loss < best_loss:
                best_loss = valid_loss
                final_test_loss = test_loss

            f.write(f'epoch {epoch + 1} eval loss: {valid_loss}\n')

        f.write(f'run {run_idx + 1} final test loss: {final_test_loss}\n\n')

    return final_test_loss


final_test_losses = []
for run_idx in range(args.runs):
    final_test_losses.append(run_experiment(run_idx))

mean_loss = float(np.mean(final_test_losses))
std_loss = float(np.std(final_test_losses, ddof=1)) if len(final_test_losses) > 1 else 0.0

summary_path = Path('result/FNH_LEM.txt')
with summary_path.open('a') as f:
    f.write(f'mean final test loss over {args.runs} runs: {mean_loss}\n')
    f.write(f'std final test loss over {args.runs} runs: {std_loss}\n\n')

print(f'mean final test loss over {args.runs} runs: {mean_loss}')
print(f'std final test loss over {args.runs} runs: {std_loss}')
