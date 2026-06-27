import json


def get_query_id(query):
    for field_name in ["query_id", "qid"]:
        if hasattr(query, field_name):
            return str(getattr(query, field_name))

    raise ValueError(f"Cannot identify query id for {query}")


def get_doc_id(doc):
    for field_name in ["doc_id", "docno"]:
        if hasattr(doc, field_name):
            return str(getattr(doc, field_name))

    raise ValueError(f"Cannot identify document id for {doc}")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
