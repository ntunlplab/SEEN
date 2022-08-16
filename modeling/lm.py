from transformers import AutoConfig, AutoModelForSequenceClassification

from modeling.base import BaseModel


class LongformerClassifier(BaseModel):
    model_name = "allenai/longformer-base-4096"

    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )

    def forward(self, input_ids, attention_mask, global_attention_mask, **_):
        return self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        ).logits


class LongformerLargeClassifier(LongformerClassifier):
    model_name = "allenai/longformer-large-4096"
