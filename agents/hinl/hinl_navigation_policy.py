import math
from typing import Optional

import einops
import torch
from torch import Tensor, nn

from .forget_attn_lstm import ForgetAttnLSTMArchived


class ForgetAttnLSTMNavigationPolicyArchived(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.gpu_id = 0

        self.n_layers = 2
        self.train_num_steps = 50
        self.policy_hidden_state_size = 512

        self.policy_inputs = ['visual_representation', 'last_action']

        self.policy = ForgetAttnLSTMArchived(
            3200, 512, 2,
        )

        self.d_model = 512
        self.tau_historical_cells = 5
        self.add_temporal_embedding = True

        self.action_embed_linear = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU()
        )

        if self.add_temporal_embedding:
            self.temporal_embedding = PositionalEncoding1D(self.d_model, max_len=1000)

        self.critic_linear = nn.Sequential(
            nn.Linear(512, 1),
        )
        
        self.actor_linear = nn.Sequential(
            nn.Linear(512, 6 // 2),
            nn.Linear(6 // 2, 6),
        )

        self.init_navigation_policy_network()


    def init_navigation_policy_network(self) -> None:
        self.init_policy_bias_as_zero(self.policy)


    def init_policy_bias_as_zero(self, policy: nn.LSTM):
        for name, param in policy.named_parameters():
            if "bias" in name:
                param.data.fill_(0)


    def prepare_inputs(self, model_input, visual_outputs: dict) -> dict:
        navigation_policy_inputs = {
            'visual_representation': visual_outputs['visual_representation'],
            'last_action': model_input.action_probs,
            'policy_hidden_states': model_input.hidden,
            'n_steps': model_input.n_step,
        }
        return navigation_policy_inputs


    def extra_policy_inputs(self, inputs: dict) -> list:
        policy_inputs = []
        if 'last_action' in self.policy_inputs:
            action_embedding = self.action_embed_linear(inputs['last_action']).unsqueeze(dim=1)
            policy_inputs.append(action_embedding)
        # TODO: add code to embed relative state? using linear or positional encoding?
        # if 'relative_state' in self.policy_inputs:
        #     state_embedding = einops.rearrange(self.relative_state_embed_linear(inputs['relative_state']), '1 c -> 1 1 c')
        #     policy_inputs.append(state_embedding)
        return policy_inputs


    def forward(self, inputs: dict) -> dict:
        with torch.cuda.device(self.gpu_id):
            n_steps = inputs['n_steps']

            if (n_steps % self.train_num_steps == 0 and n_steps > 0) and hasattr(self, 'historical_cells'):
                self.historical_cells = self.historical_cells.detach()

            if n_steps == 0:
                self.historical_cells = torch.zeros((self.tau_historical_cells, self.n_layers, self.d_model)).cuda()

            navigation_policy_inputs = [inputs['visual_representation']]
            navigation_policy_inputs.extend(self.extra_policy_inputs(inputs))
            embedding = einops.rearrange(torch.cat(navigation_policy_inputs, dim=1), '1 n c -> 1 1 (n c)')

            if self.add_temporal_embedding:
                historical_cells = self.historical_cells + einops.repeat(self.temporal_embedding.pe[
                    max(n_steps-self.tau_historical_cells, 0):max(n_steps, self.tau_historical_cells)], 'n 1 c -> n 2 c')
            else:
                historical_cells = self.historical_cells

            output, hidden_state = self.policy(embedding, historical_cells, inputs['policy_hidden_states'])

            self.historical_cells[:-1, ...] = self.historical_cells[1:, ...].clone()
            self.historical_cells[-1, ...] = hidden_state[0].squeeze(1)

            state_representation = output.reshape([1, self.policy_hidden_state_size])

            actor_out = self.actor_linear(state_representation)
            critic_out = self.critic_linear(state_representation)

            return {
                'actor_output': actor_out,
                'critic_output': critic_out,
                'policy_hidden_states': hidden_state,
                'state_representation': state_representation,
            }


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, i: Optional[int] = None, end: Optional[int] = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if i is None and end is None:
            positional_encoding = self.pe[:x.size(0)]
        elif i is not None and end is None:
            positional_encoding = self.pe[i:i+1]
        elif i is None and end is not None:
            positional_encoding = self.pe[max(end-x.size(0), 0):max(end, 1)]
        else:
            raise NotImplementedError

        x = x + positional_encoding
        return self.dropout(x)
