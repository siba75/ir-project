from text_processing import preprocess_text


def lexical_scores(
    query_tokens: list[str],
    resources,
    k1: float | None = None,
    b: float | None = None,
):
    bm25 = resources["bm25"]
    doc_ids = resources["doc_ids"]
    original_k1 = getattr(bm25, "k1", None)
    original_b = getattr(bm25, "b", None)

    try:
        if k1 is not None:
            bm25.k1 = k1

        if b is not None:
            bm25.b = b

        raw_scores = bm25.get_scores(query_tokens)
    finally:
        if original_k1 is not None:
            bm25.k1 = original_k1

        if original_b is not None:
            bm25.b = original_b

    return {
        str(doc_ids[index]): float(score)
        for index, score in enumerate(raw_scores)
        if score > 0
    }


def tokenize_query(query: str):
    return preprocess_text(query)
