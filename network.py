import torch
import torch.nn as nn
import math

class LEMCell(nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)

        return y, z

class LEM(nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.):
        super(LEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp,nhid,dt)
        self.classifier = nn.Linear(nhid, nout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        y_hidds = []
        for x in input:
            y, z = self.cell(x,y,z)
            y_hidds.append(y)
        out = self.classifier(torch.stack((y_hidds), dim=0))
        return out
    

class RNNODE(nn.Module):
    def __init__(self, input_dim=1, n_latent=128, n_hidden=128):
        super(RNNODE, self).__init__()
        self.fc1 = nn.Linear(input_dim + n_latent, n_hidden)
        self.tanh = nn.Tanh()

    def forward(self, t, h, x):
        out = self.fc1(torch.cat((x, h), dim=1))
        out = self.tanh(out)
        return out

    def initHidden(self):
        return


class OutputNN(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(OutputNN, self).__init__()
        self.fc = nn.Linear(n_latent, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.n_latent = n_latent

    def forward(self, h):
        out = self.fc(h)
        return out


class RNN(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=n_latent, batch_first=True)
        self.linear = nn.Linear(n_latent, input_dim)

    def forward(self, x, h_0):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        h_0 = h_0.reshape(1, h_0.shape[0], h_0.shape[1])

        _, final_hidden_state = self.rnn(x, h_0)
        output = self.linear(final_hidden_state)

        return output.reshape(output.shape[1], output.shape[2]), final_hidden_state.reshape(h_0.shape[1], h_0.shape[2])


class LSTM(nn.Module):
    def __init__(self, input_dim=2, n_latent=128):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_latent, batch_first=True)
        self.linear = nn.Linear(n_latent, input_dim)

    def forward(self, x, h_0, c_0):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        #         print(x.shape)
        h_0 = h_0.reshape(1, h_0.shape[0], h_0.shape[1])
        c_0 = c_0.reshape(1, c_0.shape[0], c_0.shape[1])

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        output = self.linear(output)

        #         print(output.shape)
        output = output.reshape(output.shape[0], output.shape[2])
        final_hidden_state = final_hidden_state.reshape(h_0.shape[1], h_0.shape[2])
        final_cell_state = final_cell_state.reshape(c_0.shape[1], c_0.shape[2])

        return output, final_hidden_state, final_cell_state
