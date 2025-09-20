from torch import nn, optim, Tensor
import torch
import torch.nn.utils
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import argparse

import data
from network import RNNODE, OutputNN


parser = argparse.ArgumentParser(description='training parameters (RNN-ODE to mimic LEM loop)')
parser.add_argument('--nhid', type=int, default=16, help='hidden/latent size')
parser.add_argument('--epochs', type=int, default=400, help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.00904, help='learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()
print(args)

ninp = 1
nout = 1

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Generate data exactly as in FitzHughNagumo_task (v_t -> v_{t+1})
train_x, train_y = data.get_data(128)
valid_x, valid_y = data.get_data(128)
test_x, test_y = data.get_data(1024)
print('Finished generating data')

# Datasets/loaders identical to LEM setup
train_dataset = TensorDataset(Tensor(train_x).float(), Tensor(train_y).float())
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)

valid_dataset = TensorDataset(Tensor(valid_x).float(), Tensor(valid_y).float())
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=128)

test_dataset = TensorDataset(Tensor(test_x).float(), Tensor(test_y).float())
testloader = DataLoader(test_dataset, shuffle=False, batch_size=128)

# Build RNN-ODE components
odefunc = RNNODE(input_dim=ninp, n_latent=args.nhid, n_hidden=args.nhid).to(args.device)
outputfunc = OutputNN(input_dim=nout, n_latent=args.nhid).to(args.device)

objective = nn.MSELoss()
optimizer = optim.Adam(list(odefunc.parameters()) + list(outputfunc.parameters()), lr=args.lr)


def forward_roll(x_seq):
    """
    Unroll RNN-ODE across time with simple forward Euler (dt=1),
    mirroring LEM’s per-step prediction: given x_t, predict x_{t+1}.
    - x_seq: Tensor [T, B, ninp]
    Returns: Tensor [T, B, nout]
    """
    T, B, _ = x_seq.shape
    h = torch.zeros(B, args.nhid, device=x_seq.device)
    outs = []
    for t in range(T):
        x_t = x_seq[t]
        h = h + odefunc(t=torch.tensor(0.0, device=x_seq.device), x=x_t, h=h)
        yhat = outputfunc(h)
        outs.append(yhat)
    return torch.stack(outs, dim=0)


def eval_loop(dataloader):
    odefunc.eval()
    outputfunc.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y = y.permute(1, 0, 2).to(args.device)  # [T,B,1]
            x = x.permute(1, 0, 2).to(args.device)  # [T,B,1]
            out = forward_roll(x)
            loss = torch.sqrt(objective(out, y)).item()
    return loss


best_loss = float('inf')
for epoch in range(args.epochs):
    odefunc.train()
    outputfunc.train()
    for x, y in trainloader:
        y = y.permute(1, 0, 2).to(args.device)
        x = x.permute(1, 0, 2).to(args.device)
        optimizer.zero_grad()
        out = forward_roll(x)
        loss = objective(out, y)
        loss.backward()
        optimizer.step()

    valid_loss = eval_loop(validloader)
    test_loss = eval_loop(testloader)
    if valid_loss < best_loss:
        best_loss = valid_loss
        final_test_loss = test_loss

    Path('result').mkdir(parents=True, exist_ok=True)
    with open('result/FNH_RNN_ODE.txt', 'a') as f:
        f.write('eval loss: ' + str(valid_loss) + '\n')

with open('result/FNH_RNN_ODE.txt', 'a') as f:
    f.write('final test loss: ' + str(final_test_loss) + '\n')

print('Done training RNN-ODE (LEM-style loss). Best valid sqrt(MSE):', best_loss)

