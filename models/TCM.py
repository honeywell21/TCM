import torch
import torch.nn as nn
from layers.Invertible import RevIN
from layers.Projection import ChannelProjection
from utils.decomposition import svd_denoise, NMF
# from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from layers.FourierCorrelation import FourierBlock


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class TCMBlock(nn.Module):

    def __init__(
            self,
            tokens_dim,
            channel_dim,
            dropout,
            tokens_mlp_ratio=4,
            channel_dim_ratio=4,
            mlp_layer=MLP,
            norm_layer=Affine,
            init_values=1e-4,
            channel_drop=0.5,
            drop_path=0.,
    ):
        super().__init__()
        self.tokens_hidden_dim = tokens_dim * tokens_mlp_ratio
        channel_hidden_dim = int(channel_dim * channel_dim_ratio)
        self.affine1 = norm_layer(channel_dim)
        self.affine2 = norm_layer(channel_dim)
        self.linear_tokens = mlp_layer(tokens_dim, tokens_dim * tokens_mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_channels = mlp_layer(channel_dim, channel_hidden_dim, dropout)
        self.ls1 = nn.Parameter(init_values * torch.ones(channel_dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(channel_dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.affine1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.affine2(x)))
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            TCMBlock(configs.seq_len, configs.enc_in, configs.dropout) for _ in range(configs.e_layers)
        ])
        print(configs.dropout)
        self.norm = nn.LayerNorm(configs.enc_in) if configs.norm else None
        self.projection = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        # self.refine = Mlp(configs.pred_len, configs.d_model) if configs.refine else None
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.affine = Affine(configs.enc_in)

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        for block in self.mlp_blocks:
            x = block(x)
        x = self.projection(x)
        x = self.rev(x, 'denorm') if self.rev else x
        return x
