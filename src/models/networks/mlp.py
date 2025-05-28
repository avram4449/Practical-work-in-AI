import torch
import torch.nn as nn
from models.networks.registry import register_model


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dims=None,
        non_lin=nn.ReLU,
        dropout_p=0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 1024, 512, 256]

        layers = []
        last_dim = dim_in
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(non_lin())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            last_dim = h

        layers.append(nn.Linear(last_dim, dim_out))
        # No LogSoftmax here, output raw logits

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@register_model
def fcnet(config, **kwargs):
    dim_in = config["data"]["shape"][-1]
    dim_out = config["data"]["num_classes"]

    return MLP(
        dim_in=dim_in,
        dim_out=dim_out,
        hidden_dims=[1024, 1024, 512, 256],
        non_lin=nn.ReLU,
        dropout_p=config["model"].get("dropout", 0.0)
    )
