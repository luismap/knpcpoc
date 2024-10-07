from typing import List
import requests
import json
import numpy as np
from core.utils.ModelsCml import get_model_access_key, MODEL_API_URL
from sentence_transformers import CrossEncoder

# RERANKER_SRVC_KEY = "reranker"


class Reranker:
    AICHAT_SRVC_NAME = "aichat"
    QEXPANSION_SRVC_NAME = "qexpansion"
    VDB_SRVC_NAME = "vdb"
    RERANKER_SRVC_NAME = "reranker"

    AICHAT_SRVC_KEY = get_model_access_key(AICHAT_SRVC_NAME)
    QEXPANSION_SRVC_KEY = get_model_access_key(QEXPANSION_SRVC_NAME)
    VDB_SRVC_KEY = get_model_access_key(VDB_SRVC_NAME)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def __init__(self):
        return None

    def create_context(docs):
        """given the list unique docs reranked.
        return a context for the llm
        """

    def get_idx_top_rank(self, scores):
        rank = np.argsort(scores)[::-1]
        return rank

    def get_top_n_docs(self, ids, unique_docs, n: int = 3):
        docs = []
        for i in range(0, n):
            docs.append(unique_docs[ids[i]])
        return docs

    def question_doc_pairs(self, orginal_question: str, unique_docs):
        pairs = []
        for t in unique_docs:
            pairs.append((orginal_question, t[1]))
        return pairs

    def clean_query_list(self, questions: List[str]) -> List[str]:
        removed_empty_items = filter(lambda x: x != "", questions)
        return list(removed_empty_items)

    def flatten_received_docs(self, results, embeddings: bool = False):
        if embeddings:
            flatten_retrieved_documents = [
                (idx, doc, meta, embed)
                for tuples in list(
                    zip(
                        results["ids"],
                        results["docs"],
                        results["metas"],
                        results["embeddings"],
                    )
                )
                for idx, doc, meta, embed in zip(*tuples)
            ]
        else:
            flatten_retrieved_documents = [
                (idx, doc, meta)
                for tuples in list(
                    zip(results["ids"], results["docs"], results["metas"])
                )
                for idx, doc, meta in zip(*tuples)
            ]
        return flatten_retrieved_documents

    def deduplicated_retrieved_docs(self, docs):
        cache = set()
        flatten_unique_retrieved_documents = []

        for t in docs:
            if t[0] not in cache:
                flatten_unique_retrieved_documents.append(t)
                cache.add(t[0])
        return flatten_unique_retrieved_documents

    def rerank(self, args):
        """given a question and number of n_gen docs to be create for each query.
        do query expansion,then rerank and choose the best n docs.
        return the top n docs
        ex.
        {"question": "what is FCA", "n_gen":3, "n_doc": 3,
        "top_n":4, "embeddings": 0}
        return a list of.
        [{"idx": "doc index", "doc": "document", "metadata": "metadat of the document" }]

        """
        query = args["question"]
        n_gen = args["n_gen"]
        n_docs = args["n_doc"]
        top_n = args["top_n"]
        embeddings = args["embeddings"]

        # query expansion
        url_qe = f"{MODEL_API_URL}?accessKey={self.QEXPANSION_SRVC_KEY}"
        qe_payload = {"request": {"question": query, "n_gen": n_gen}}
        data = json.dumps(qe_payload)
        headers = {"Content-Type": "application/json"}
        eq_rsp = requests.post(url_qe, data=data, headers=headers)
        expanded_query = eq_rsp.json()["response"]["answer"]  # get from expansion srvc

        query_list = self.clean_query_list(expanded_query.split("\n"))
        query_list.append(query)

        # docs retrieval
        url_vdb = f"{MODEL_API_URL}?accessKey={self.VDB_SRVC_KEY}"
        vdb_payload = {
            "request": {
                "question": query_list,
                "n_results": n_docs,
                "embeddings": embeddings,
            }
        }
        data_vdb = json.dumps(vdb_payload)
        headers_vdb = {"Content-Type": "application/json"}
        vdb_rsp = requests.post(url_vdb, data=data_vdb, headers=headers_vdb)
        docs = vdb_rsp.json()["response"]  # get from expansion srvc
        # print(docs.keys())
        # print(docs)
        print(len(docs))
        flattened_docs = self.flatten_received_docs(docs)
        print(len(flattened_docs))
        unique_docs = self.deduplicated_retrieved_docs(flattened_docs)
        print(len(unique_docs))
        pairs = self.question_doc_pairs(query, unique_docs=unique_docs)
        print(pairs)
        scores = self.cross_encoder.predict(pairs)
        print(scores)
        best_docs_idx = self.get_idx_top_rank(scores)
        best_docs = self.get_top_n_docs(best_docs_idx, unique_docs, top_n)
        return best_docs
