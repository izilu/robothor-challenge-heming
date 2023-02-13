import einops
import math
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn.parameter import Parameter


class ForgetAttnLSTMArchived(nn.Module):
    def __init__(
        self,
        input_sz,
        hidden_sz,
        n_layers,
    ):

        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.n_layers = n_layers

        # self.configs = cfg

        self.i_i = nn.ModuleList([nn.Linear(input_sz, hidden_sz) if i == 0 else nn.Linear(hidden_sz, hidden_sz) for i in range(n_layers)])
        self.i_f = nn.ModuleList([nn.Linear(input_sz, hidden_sz) if i == 0 else nn.Linear(hidden_sz, hidden_sz) for i in range(n_layers)])
        self.i_g = nn.ModuleList([nn.Linear(input_sz, hidden_sz) if i == 0 else nn.Linear(hidden_sz, hidden_sz) for i in range(n_layers)])

        self.h_i = nn.ModuleList([nn.Linear(hidden_sz, hidden_sz) for _ in range(n_layers)])
        self.h_f = nn.ModuleList([nn.Linear(hidden_sz, hidden_sz) for _ in range(n_layers)])
        self.h_g = nn.ModuleList([nn.Linear(hidden_sz, hidden_sz) for _ in range(n_layers)])

        forget_layer = nn.TransformerDecoderLayer(d_model=512, nhead=4)

        self.forgets = nn.ModuleList([
            nn.TransformerDecoder(forget_layer, num_layers=1) for _ in range(n_layers)
        ])

        self.init_weights()


    def init_parameters(self, name):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for parameter in getattr(self, name):
            init.uniform_(parameter, -stdv, stdv)


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


    def forward(self, x, historical_states, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        gpu_device = x.device

        hidden_seq = []
        if init_states is None:
            h_t_1, c_t_1 = (torch.zeros(self.n_layers, bs, self.hidden_size).to(gpu_device),
                            torch.zeros(self.n_layers, bs, self.hidden_size).to(gpu_device))
        else:
            h_t_1, c_t_1 = init_states

        HS = self.hidden_size
        c_t = []
        h_t = []

        for t in range(seq_sz):
            c_t = []
            h_t = []

            x_i_t, x_f_t, x_g_t = x[:, t, :], x[:, t, :], x[:, t, :]

            # batch the computations into a single matrix multiplication
            for i in range(self.n_layers):
                if i == 0:
                    i_t = torch.sigmoid(self.i_i[i](x_i_t) + self.h_i[i](h_t_1[i, ...]))
                    f_t = torch.sigmoid(self.i_f[i](x_f_t) + self.h_f[i](h_t_1[i, ...]))
                    g_t = torch.tanh(self.i_g[i](x_g_t) + self.h_g[i](h_t_1[i, ...]))

                else:
                    i_t = torch.sigmoid(self.i_i[i](x_i_t) + self.h_i[i](h_t_1[i, ...]))
                    f_t = torch.sigmoid(self.i_f[i](x_i_t) + self.h_f[i](h_t_1[i, ...]))
                    g_t = torch.tanh(self.i_g[i](x_i_t) + self.h_g[i](h_t_1[i, ...]))

                f_t = torch.sigmoid(c_t_1[i, ...] - (self.forgets[i](f_t[:, None, :], historical_states[:, i, None, ...])).squeeze(1))

                c_i_t = f_t * c_t_1[i, ...] + i_t * g_t
                x_i_t = torch.tanh(c_i_t)

                c_t.append(einops.rearrange(c_i_t, '1 c -> 1 1 c'))
                h_t.append(einops.rearrange(x_i_t, '1 c -> 1 1 c'))

            hidden_seq.append(h_t[-1])
        hidden_seq = torch.cat(hidden_seq, dim=0)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_t = torch.cat(c_t, dim=0).contiguous()
        h_t = torch.cat(h_t, dim=0).contiguous()

        return hidden_seq, (h_t, c_t)