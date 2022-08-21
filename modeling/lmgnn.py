import logging
import os
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerModel

from modeling.layers import MLP, Attn2NodeFeature, BaseModule
from modeling.lm import LongformerClassifier, LongformerLargeClassifier

logger = logging.getLogger()


class LongformerGATEncoder(BaseModule):
    def __init__(
        self,
        config: LongformerConfig,
        layer: nn.ModuleList,
        fusion_with_graph_layers: int = 5,
        return_attention_weights=None,
    ):
        super().__init__()
        self.return_attention_weights = return_attention_weights
        self.config = config
        self.layer = layer
        self.fusion_with_graph_layers = fusion_with_graph_layers
        self.gelu = nn.GELU()
        self.gnn_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=config.hidden_size,
                    out_channels=config.hidden_size // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    dropout=config.attention_probs_dropout_prob,
                )
                for _ in range(fusion_with_graph_layers)
            ]
        )
        self.node_embed_layers = nn.ModuleList(
            [Attn2NodeFeature(config.hidden_size) for _ in range(fusion_with_graph_layers)]
        )
        self.fusion_layers = nn.ModuleList(
            [
                MLP(
                    config.hidden_size * 2,
                    config.hidden_size * 2,
                    config.hidden_size * 2,
                    2,
                    config.hidden_dropout_prob,
                )
                for _ in range(fusion_with_graph_layers - 1)
            ]
        )

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: torch.tensor,
        article_node2token: List[torch.tensor],
        article_node2token_mask: List[torch.tensor],
        graph: Batch,
        super_node_idx: torch.tensor,
        padding_len: int,
        **_,
    ):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        gat_attns = []

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )[0]

            if padding_len > 0:
                # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
                hidden_states = hidden_states[:, :-padding_len]

            if i < self.config.num_hidden_layers - self.fusion_with_graph_layers:
                continue

            # GNN
            fusion_layer_index = i - self.config.num_hidden_layers + self.fusion_with_graph_layers
            graph.x = torch.vstack(
                self.node_embed_layers[fusion_layer_index](
                    hidden_states, article_node2token, article_node2token_mask
                )
            )
            gnn_output = self.gnn_layers[fusion_layer_index](
                graph.x, graph.edge_index, return_attention_weights=self.return_attention_weights
            )
            if self.return_attention_weights:
                graph.x, gat_attn = gnn_output
                gat_attn[0].detach().cpu()
                gat_attn[1].detach().cpu()
                gat_attns.append(gat_attn)
            else:
                graph.x = gnn_output
            if i == self.config.num_hidden_layers - 1:
                break
            graph.x = self.gelu(graph.x)
            graph.x = F.dropout(graph.x, self.config.hidden_dropout_prob, training=self.training)

            # Exchange info between LM and GNN hidden states (Modality interaction)
            x = graph.x[super_node_idx]
            x = torch.cat([hidden_states[:, 0, :], x], dim=1)
            x = self.fusion_layers[fusion_layer_index](x)
            lm_cls, gnn_cls = torch.split(x, [hidden_states.size(2), graph.x.size(1)], dim=1)
            hidden_states[:, 0, :] = lm_cls
            graph.x[super_node_idx] = gnn_cls.to(dtype=graph.x.dtype)

        if self.return_attention_weights:
            return hidden_states, graph, gat_attns
        else:
            return hidden_states, graph


class LongformerGAT(BaseModule, ModuleUtilsMixin):
    def __init__(self, fusion_with_graph_layers: int = 5, pretrained_path=None):
        super().__init__()
        model_name = LongformerClassifier.model_name
        self.config = AutoConfig.from_pretrained(model_name)
        if pretrained_path:
            model = LongformerClassifier.load_from_checkpoint(pretrained_path).classifier.longformer
        else:
            model: LongformerModel = AutoModel.from_pretrained(
                model_name, config=self.config, add_pooling_layer=False
            )
        self.embeddings = model.embeddings
        self.encoder = LongformerGATEncoder(
            config=self.config,
            fusion_with_graph_layers=fusion_with_graph_layers,
            layer=model.encoder.layer,
        )

    def _pad_to_window_size(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens

        return padding_len, input_ids, attention_mask

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor
    ):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        global_attention_mask: torch.tensor,
        article_node2token: List[torch.tensor],
        article_node2token_mask: List[torch.tensor],
        graph: Batch,
        super_node_idx: torch.tensor,
        **kwargs,
    ):
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.device)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )[:, 0, 0, :]

        for k, v in kwargs.items():
            if not k.endswith("_mask") or "node2token" in k or v is None:
                continue
            if global_attention_mask is not None:
                kwargs[k] = self._merge_to_attention_mask(v, global_attention_mask)

            _, _, kwargs[k] = self._pad_to_window_size(
                input_ids=input_ids, attention_mask=kwargs[k], pad_token_id=self.config.pad_token_id
            )
            kwargs[k] = self.get_extended_attention_mask(kwargs[k], input_shape, self.device)[
                :, 0, 0, :
            ]

        embedding_output = self.embeddings(input_ids=input_ids)
        return self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            article_node2token=article_node2token,
            article_node2token_mask=article_node2token_mask,
            graph=graph,
            super_node_idx=super_node_idx,
            padding_len=padding_len,
            **kwargs,
        )


class LongformerLargeGAT(LongformerGAT):
    def __init__(self, fusion_with_graph_layers: int = 5, pretrained_path=None):
        super().__init__(fusion_with_graph_layers=fusion_with_graph_layers)
        model_name = LongformerLargeClassifier.model_name
        self.config = AutoConfig.from_pretrained(model_name)
        if pretrained_path:
            model = LongformerLargeClassifier.load_from_checkpoint(
                pretrained_path
            ).classifier.longformer
        else:
            model: LongformerModel = AutoModel.from_pretrained(
                model_name, config=self.config, add_pooling_layer=False
            )
        self.embeddings = model.embeddings
        self.encoder = LongformerGATEncoder(
            config=self.config,
            fusion_with_graph_layers=fusion_with_graph_layers,
            layer=model.encoder.layer,
            return_attention_weights=True if os.getenv("RETURN_ATTENTION_WEIGHTS") == "1" else None,
        )
