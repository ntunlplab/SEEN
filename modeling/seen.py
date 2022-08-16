import logging
from typing import List

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from modeling.base import SEENBaseModel
from modeling.layers import Classifier
from modeling.lmgnn import LongformerGAT, LongformerLargeGAT

logger = logging.getLogger()


class SEENLongformer(SEENBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LongformerGAT()
        self.pre_retold_classifier = Classifier(self.encoder.config.hidden_size * 3, 1)
        self.post_retold_classifier = Classifier(self.encoder.config.hidden_size * 3, 3)
        self.related_node_classifier = Classifier(self.encoder.config.hidden_size, 1)

    def calculate_related_loss(self, related_node_logits, related_nodes):
        assert len(related_node_logits) == len(related_nodes)
        related_node_loss = None
        for logit, related in zip(related_node_logits, related_nodes):
            if (related > 0).any():
                if not related_node_loss:
                    related_node_loss = self.related_node_ce_fn(logit.reshape(-1), related.float())
                else:
                    related_node_loss += self.related_node_ce_fn(logit.reshape(-1), related.float())
        if related_node_loss is None:
            return torch.tensor(0.0, device=self.device)
        else:
            return related_node_loss / len(related_nodes)

    def forward(
        self,
        input_ids: torch.tensor,
        mask: torch.tensor,
        global_mask: torch.tensor,
        article_node2token: List[torch.tensor],
        article_node2token_mask: List[torch.tensor],
        graph: Batch,
        super_node_idx: torch.tensor,
        **kwargs,
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=mask,
            global_attention_mask=global_mask,
            article_node2token=article_node2token,
            article_node2token_mask=article_node2token_mask,
            graph=graph,
            super_node_idx=super_node_idx,
            **kwargs,
        )
        embed, graph = output[:2]
        graph_embedding = global_mean_pool(graph.x, graph.batch)
        x = torch.concat([embed[:, 0], graph.x[super_node_idx], graph_embedding], dim=1)
        return (
            self.pre_retold_classifier(x),
            self.post_retold_classifier(x),
            self.related_node_classifier(graph.x),
        )


class SEENLongformerLarge(SEENLongformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LongformerLargeGAT()
        self.pre_retold_classifier = Classifier(self.encoder.config.hidden_size * 3, 1)
        self.post_retold_classifier = Classifier(self.encoder.config.hidden_size * 3, 3)
        self.related_node_classifier = Classifier(self.encoder.config.hidden_size, 1)


class SEENLongformerPretrained(SEENLongformer):
    def __init__(self, *args, pretrained_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LongformerGAT(pretrained_path=pretrained_path)


class SEENLongformerLargePretrained(SEENLongformerLarge):
    def __init__(self, *args, pretrained_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LongformerLargeGAT(pretrained_path=pretrained_path)


if __name__ == "__main__":
    model1 = SEENLongformerPretrained()
    model2 = SEENLongformerLargePretrained()
