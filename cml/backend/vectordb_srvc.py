from core.vectordbs.MyChroma import MyChroma
import sys

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


mycvdb = MyChroma()

folder_path = "data/pdf/"

pdf_docs = mycvdb.read_pdf_langchain_docs(folder_path)

load = True

store = mycvdb.load_chroma_docs(pdf_docs, "myknpc11")


def vdb_query(args):
    question = args["question"]
    try:
        results = args["n_results"]
    except KeyError:
        results = 3

    responses = store.query(
        query_texts=question,
        n_results=results,
        include=["metadatas", "documents", "distances", "embeddings"],
    )

    return {"docs": responses["documents"], "metas": responses["metadatas"]}
