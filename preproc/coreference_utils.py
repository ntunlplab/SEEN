from typing import List

from allennlp.predictors.predictor import Predictor


def load_model():
    allen_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    return Predictor.from_path(allen_url, cuda_device=1)


def get_span_noun_indices(doc, cluster: List[List[int]]):
    spans = [doc[span[0] : span[1] + 1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    spans_tag = [[token.tag_ for token in span] for span in spans]
    span_noun_indices = [
        i
        for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ["NOUN", "PROPN"])
    ]
    if not span_noun_indices:
        span_noun_indices = [
            i
            for i, (span_pos, span_tag, span) in enumerate(zip(spans_pos, spans_tag, spans))
            if len(span_pos) == len(span_tag) == 1
            and span_pos[0] == "PRON"
            and span_tag[0] == "PRP"
            and span.text.lower() not in {"me", "him", "her", "us", "them"}
        ]
    if not span_noun_indices:
        span_noun_indices = [
            i
            for i, (span_pos, span_tag, span) in enumerate(zip(spans_pos, spans_tag, spans))
            if len(span_pos) == len(span_tag) == 1
            and span_pos[0] == "PRON"
            and span_tag[0] == "PRP"
        ]
    return span_noun_indices


def get_cluster_head_idx(doc, cluster):
    noun_indices = get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0


def get_clusters(doc, clusters):
    def get_span(span, allen_document):
        start = span[0]
        while any(
            " ".join(i.text for i in allen_document[start : span[1] + 1]).startswith(i)
            for i in {"a ", "an ", "the ", "to ", "few ", "some ", "lot ", "lots "}
        ):
            start += 1
        return allen_document[start : span[1] + 1]

    allen_document = [t for t in doc]
    results = []
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            results.append(
                (
                    get_span(cluster_head, allen_document),
                    [get_span(span, allen_document) for span in cluster],
                )
            )
    return results
