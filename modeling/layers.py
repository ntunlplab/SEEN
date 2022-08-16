from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning.core.mixins.device_dtype_mixin import DeviceDtypeModuleMixin
from transformers import AutoConfig, AutoModel
from transformers.models.longformer.modeling_longformer import (
    LongformerBaseModelOutputWithPooling,
    LongformerModel,
)


class BaseModule(DeviceDtypeModuleMixin):
    pass


class LanguageModel(BaseModule):
    LM_MODEL_NAME = "allenai/longformer-base-4096"

    def __init__(self, fixed_lm: bool = False, add_pooling_layer=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(self.LM_MODEL_NAME)
        self.lm: LongformerModel = AutoModel.from_pretrained(
            self.LM_MODEL_NAME, config=self.config, add_pooling_layer=add_pooling_layer
        )
        if fixed_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        global_mask = torch.zeros(4096, dtype=torch.long, device=self.device, requires_grad=False)
        global_mask[0] = 1
        self.register_buffer("global_mask", global_mask)

    def forward(
        self, input_ids: torch.tensor, mask: torch.tensor, global_mask=None, **_
    ) -> LongformerBaseModelOutputWithPooling:
        if global_mask is None:
            global_mask = self.global_mask
        global_mask = global_mask[: input_ids.size(1)]
        return self.lm(input_ids=input_ids, attention_mask=mask, global_attention_mask=global_mask)

    def to(self, *args: Any, **kwargs: Any):
        self.global_mask.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class Attn2NodeFeature(BaseModule):
    def __init__(self, d_model: int, attn_layers=1, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attns = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=8, batch_first=True, dropout=dropout, norm_first=True
                )
                for _ in range(attn_layers)
            ]
        )
        # padding of max number of node: 200
        self.register_buffer(
            "padding", torch.zeros(300, 1, device=self.device, dtype=torch.int), persistent=False
        )

    def forward(
        self,
        embed: torch.tensor,
        node2token: List[torch.tensor],
        node2token_mask: List[torch.tensor],
    ) -> List[torch.tensor]:
        node_embed = []
        for emb, n2t, n2t_mask in zip(embed, node2token, node2token_mask):
            padding = self.padding[: len(n2t)]
            n2t = torch.hstack([padding, n2t])
            n2t_mask = torch.hstack([padding, n2t_mask])
            emb = emb[n2t, :]
            for attn in self.attns:
                emb = attn(src=emb, src_key_padding_mask=n2t_mask)
            node_embed.append(emb[:, 0])
        return node_embed


class Classifier(BaseModule):
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """

    activation_classes = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        batch_norm=False,
        init_last_layer_bias_to_zero=False,
        layer_norm=False,
        activation="gelu",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f"{i}-Linear", nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f"{i}-Dropout", nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f"{i}-BatchNorm1d", nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f"{i}-LayerNorm", nn.LayerNorm(self.hidden_size))
                self.layers.add_module(
                    f"{i}-{activation}", self.activation_classes[activation.lower()]()
                )
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)
