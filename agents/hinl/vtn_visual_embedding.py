from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn

from .transformer import VisualTransformer


class VTRGBVisualEmbedding(nn.Module):
    """RGB visual embedding for VT
    """
    def __init__(self) -> None:
        # sourcery skip: merge-duplicate-blocks, remove-redundant-if
        super().__init__()

        self.n_categories = 13
        self.detection_algorithm = 'detr'

        self.image_size = 300
        self.resnet_embedding_sz = 512

        self.local_feature_output_size = 249
        self.local_feature_forward = self.local_rgb_forward

        # local visual representation learning networks
        if self.detection_algorithm == 'detr':
            self.detection_feature_input_size = 256
        elif self.detection_algorithm == 'fasterrcnn_org':
            self.detection_feature_input_size = 512
        elif self.detection_algorithm == 'fasterrcnn':
            self.detection_feature_input_size = 256
        else:
            raise RuntimeError('Unkown detection algorithm')

        self.n_local_project_layers = 1

        if self.n_local_project_layers == 1:
            self.local_project_linear = nn.Sequential(
                nn.Linear(self.detection_feature_input_size, self.local_feature_output_size),
                nn.ReLU(),
            )
        elif self.n_local_project_layers == 2:
            self.local_project_linear = nn.Sequential(
                nn.Linear(self.detection_feature_input_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.local_feature_output_size),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError

        # global visual representation learning networks
        self.global_feature_forward = self.global_rgb_forward

        self.global_feature_project_conv = nn.Sequential(
            nn.Conv2d(self.resnet_embedding_sz, 256, 1),
            nn.ReLU()
        )
        self.global_pos_embedding = get_2d_positional_embedding(7, 128)

        self.visual_transformer = VisualTransformer(
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
        )

        self.visual_representation_project_linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0)
        )

        # initialize parameters of the network
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.global_feature_project_conv[0].weight.data.mul_(relu_gain)


        # TODO: remove this later
        # self.losses = ['AdvantagePolicyLoss', 'ValueLoss']


    def update_requirements(self, requirements) -> None:
        requirements.model_requirements['input']['image_feature'] = True
        requirements.model_requirements['input']['detection_feature'] = True
        requirements.model_requirements['input']['image_feature'] = True

    def visual_forward_input(self, model_input):
        return {
            'state': model_input.state,
            'detection_inputs': model_input.detection_inputs,
        }

    def local_rgb_forward(self, visual_inputs):
        local_features = self.local_project_linear(
            visual_inputs['detection_inputs']['features'])
        local_features = torch.cat((
            local_features,                                                 # (batch_size, n_local_feature_dim, 249)
            visual_inputs['detection_inputs']['labels'],                    # (batch_size, n_local_feature_dim, 1)
            visual_inputs['detection_inputs']['bboxes'] / self.image_size,  # (batch_size, n_local_feature_dim, 4)
            visual_inputs['detection_inputs']['scores'],                    # (batch_size, n_local_feature_dim, 1)
            visual_inputs['detection_inputs']['indicator']                  # (batch_size, n_local_feature_dim, 1)
        ), dim=-1)

        return local_features                                               # (batch_size, n_local_feature_dim, 256)

    def global_rgb_forward(self, visual_inputs):
        global_rgb_features = self.global_feature_project_conv(visual_inputs['state'])
        global_features = global_rgb_features + self.global_pos_embedding.to(global_rgb_features.device)
        global_features = global_features.reshape(visual_inputs['state'].shape[0], -1, 49)

        return global_features

    def pretrain_forward(self, visual_inputs: dict):
        batch_size = visual_inputs['detection_inputs']['features'].shape[0]

        visual_inputs = {
            'state': visual_inputs['image_resnet18_feature'],
            'detection_inputs': visual_inputs['detection_inputs'],
        }

        local_features = self.local_feature_forward(visual_inputs)
        global_features = self.global_feature_forward(visual_inputs)

        visual_representation, _ = self.visual_transformer(src=local_features, query_embed=global_features)
        visual_representation = self.visual_representation_project_linear(visual_representation).reshape(batch_size, -1)

        return visual_representation

    def forward(self, visual_inputs: dict) -> dict:
        local_feature = self.local_feature_forward(visual_inputs)
        global_feature = self.global_feature_forward(visual_inputs)

        visual_representation_, _ = self.visual_transformer(src=local_feature, query_embed=global_feature)
        visual_representation = self.visual_representation_project_linear(visual_representation_)

        # visual_representation = torch.cat((visual_representation, action_embedding), dim=1).reshape(1, -1)
        return {
            'visual_representation': visual_representation,
            'global_feature': global_feature,
            # 'vision_features': visual_representation_,
        }


def get_2d_positional_embedding(size_feature_map, c_pos_embedding, gpu_id=None) -> torch.Tensor:
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    # dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)
    dim_t = 10000 ** torch.div(2 * torch.div(dim_t, 2, rounding_mode='trunc'), c_pos_embedding, rounding_mode='trunc')

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weights_init_conv(m)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        if hasattr(m.weight, 'data'):
            m.weight.data.uniform_(-w_bound, w_bound)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_conv(m):
    weight_shape = list(m.weight.data.size())
    fan_in = np.prod(weight_shape[1:4])
    fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0)


