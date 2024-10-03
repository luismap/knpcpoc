import sys

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from core.vectordbs.MyChroma import MyChroma  # noqa: E402

mycvdb = MyChroma()

folder_path = "data/pdf/"

pdf_docs = mycvdb.read_pdf_langchain_docs(folder_path)

load = True

store = mycvdb.load_chroma_docs(pdf_docs, "myknpc11")


def vdb_query(args):
    """given a query return a simililarity search on it.
    ex.
    {"question": "hi", "n_results": 3, "embeddings": 1}
    {"ids": "","docs": "", "metas": "", "embeddings": ""}

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    question = args["question"]
    try:
        results = args["n_results"]
    except KeyError:
        results = 3

    try:
        return_embeddings = args["embeddings"]
    except KeyError:
        return_embeddings = 0

    responses = store.query(
        query_texts=question,
        n_results=results,
        include=["metadatas", "documents", "distances", "embeddings"],
    )
    if return_embeddings == 0:
        rsp = {"docs": responses["documents"], "metas": responses["metadatas"]}
    else:
        rsp = {
            "ids": responses["ids"],
            "docs": responses["documents"],
            "metas": responses["metadatas"],
            "embeddings": responses["embeddings"],
        }
    return rsp
