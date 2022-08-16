from abc import ABC

import numpy as np
import pandas as pd
from flatdict import FlatDict
from sklearn.metrics._classification import (
    _check_targets,
    precision_recall_fscore_support,
    unique_labels,
)
from torchmetrics import Metric as BaseMetric

from utils.constants import PostRetoldLabel, PreRetoldLabel


def classification_report(
    y_true,
    y_pred,
    *,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
):

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if target_names is None:
        target_names = [str(label) for label in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro"):
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))
        else:
            if line_heading == "accuracy":
                row_fmt_accuracy = (
                    "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
                )
                report += row_fmt_accuracy.format(
                    line_heading, "", "", *avg[2:], width=width, digits=digits
                )
            else:
                report += row_fmt.format(line_heading, *avg, width=width, digits=digits)

    if output_dict:
        report_dict["accuracy"] = report_dict["accuracy"]["precision"]
        return report_dict
    else:
        return report


def get_report(y_label, y_pred, *, labels=None, prefix=""):
    return {
        prefix + k.replace(" avg", "").replace("-", ""): float(v)
        for k, v in FlatDict(
            classification_report(
                y_label, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
            delimiter="_",
        ).items()
    }


class Metric(BaseMetric, ABC):
    def __init__(self, *args, prefix="", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prefix = prefix
        for key in self.keys:
            self.add_state(key, default=list())

    def items(self):
        return self.compute().items()

    def _apply(self, *args, **kwargs):
        return


class NIRMetric(Metric):
    MAIN_EVAL_METRIC = "NIR_overall_macro_f1score"
    keys = ("pair_id", "story_type", "event_id", "label", "pred")
    LABEL_ENUM = {"pre-retold": PreRetoldLabel, "post-retold": PostRetoldLabel}

    @classmethod
    def get_metric_df(cls, predictions):
        df = pd.DataFrame(predictions)
        assert all(i in df.columns for i in cls.keys)

        df["NIR_label"] = df.apply(
            lambda row: cls.LABEL_ENUM[row["story_type"]](row["label"]).name, axis=1
        )
        df["NIR_pred"] = df.apply(
            lambda row: cls.LABEL_ENUM[row["story_type"]](row["pred"]).name, axis=1
        )
        return df

    @classmethod
    def get_metric(cls, df, prefix=""):
        is_pre_retold = df["story_type"] == "pre_retold"
        is_retold = ~is_pre_retold
        metric = get_report(
            y_label=df["NIR_label"],
            y_pred=df["NIR_pred"],
            labels=PreRetoldLabel.names() + PostRetoldLabel.names(),
            prefix=f"{prefix}NIR_overall_",
        )
        metric.update(
            get_report(
                y_label=df["NIR_label"][is_pre_retold],
                y_pred=df["NIR_pred"][is_pre_retold],
                labels=PreRetoldLabel.names(),
                prefix=f"{prefix}NIR_pre_retold_",
            )
        )
        metric.update(
            get_report(
                y_label=df["NIR_label"][is_retold],
                y_pred=df["NIR_pred"][is_retold],
                labels=PostRetoldLabel.names(),
                prefix=f"{prefix}NIR_post_retold_",
            )
        )
        return metric

    def update(self, pair_id, story_type, event_id, label, pred):
        self.pair_id += pair_id
        self.story_type += story_type
        self.event_id += event_id
        self.label += label
        self.pred += pred

    def compute(self):
        df = self.get_metric_df(
            [
                {
                    "pair_id": self.pair_id[i],
                    "story_type": self.story_type[i],
                    "event_id": self.event_id[i],
                    "label": self.label[i],
                    "pred": self.pred[i],
                }
                for i in range(len(self.pair_id))
            ]
        )
        return self.get_metric(df, prefix=self.prefix)


class NIRRelatedNodeMetric(NIRMetric):
    keys = (
        "pair_id",
        "story_type",
        "event_id",
        "label",
        "pred",
        "related_node",
        "related_node_pred",
        "num_nodes",
    )
    RELATED_LABELS = [
        PreRetoldLabel.unforgotten.name,
        PostRetoldLabel.consistent.name,
        PostRetoldLabel.inconsistent.name,
    ]

    @classmethod
    def get_metric_df(cls, predictions):
        df = super(NIRRelatedNodeMetric, NIRRelatedNodeMetric).get_metric_df(predictions)
        related_node_report = df.apply(
            lambda row: (
                get_report(row["related_node"], row["related_node_pred"], labels=[0, 1])
                if row["NIR_label"] in cls.RELATED_LABELS
                else np.nan
            ),
            axis=1,
        )
        for m in ["precision", "recall", "f1score", "support"]:
            for n in ["0", "1", "macro", "weighted"]:
                df[f"related_node_report_{n}_{m}"] = related_node_report.apply(
                    lambda x: x[f"{n}_{m}"] if pd.notnull(x) else np.nan
                )
        df["related_node_report_accuracy"] = related_node_report.apply(
            lambda x: x["accuracy"] if pd.notnull(x) else np.nan
        )
        df = df.fillna(0)
        return df

    @classmethod
    def get_metric(cls, df, prefix=""):
        metric = super(NIRRelatedNodeMetric, NIRRelatedNodeMetric).get_metric(df, prefix=prefix)
        metric.update(
            {
                f"{prefix}{key}": df[key][df["NIR_label"].isin(cls.RELATED_LABELS)].mean()
                for key in df.columns
                if key.startswith("related_node_report")
            }
        )
        return metric

    def update(
        self, pair_id, story_type, event_id, num_nodes, label, pred, related_node, related_node_pred
    ):
        self.pair_id += pair_id
        self.story_type += story_type
        self.event_id += event_id
        self.num_nodes += num_nodes
        self.label += label
        self.pred += pred
        self.related_node += related_node
        self.related_node_pred += related_node_pred

    def compute(self):
        indices = [0]
        for num in self.num_nodes:
            indices.append(num + indices[-1])
        indices = indices[1:-1]
        self.related_node = np.array_split(np.array(self.related_node), indices)
        self.related_node_pred = np.array_split(np.array(self.related_node_pred), indices)
        df = self.get_metric_df(
            [{key: getattr(self, key)[i] for key in self.keys} for i in range(len(self.pair_id))]
        )
        return self.get_metric(df, prefix=self.prefix)
