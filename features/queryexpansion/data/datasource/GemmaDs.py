from typing import List
from features.queryexpansion.data.datasource.api.QueryExpansion import QueryExpansion

from langchain_community.llms import Ollama


class GemmaDs(QueryExpansion):
    def __init__(self, model_id: str = "gemma2:9b"):
        self.model_id = model_id
        self.llm = Ollama(model=model_id)
        print(self.llm.invoke("hi"))

    def _clean_response(self, questions: List[str]) -> List[str]:
        removed_empty_items = filter(lambda x: x != "", questions)
        return list(removed_empty_items)

    def query(self, question: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful expert assistant on the energy sector. Your users are asking questions differents reports.
             Suggest up to five additional related questions to help them find the information they need, for the provided question.
            Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
            Make sure they are complete questions, and that they are related to the original question.
            Output one question per line. Do not number the questions.""",
            },
            {"role": "user", "content": question},
        ]

        response = self.llm.invoke(messages)
        response = self._clean_response(response.split("\n"))
        return response
