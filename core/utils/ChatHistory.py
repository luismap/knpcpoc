from typing import Dict, List


class ChatHistory:
    def __init__(self) -> None:
        self._users = {"default"}
        self._history = {"default": []}

    def add_user(self, user_id: str):
        if user_id not in self._users:
            self._users.add(user_id)
            self._history[user_id] = []
            return True
        else:
            return False

    def add_history(self, user_id: str, question: str, ai_response: Dict[str, str]):
        if user_id not in self._users:
            self.add_user(user_id)
        question_parse = {"role": "user", "content": question}
        self._history[user_id].append(question_parse)
        self._history[user_id].append(ai_response)

    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        if user_id not in self._users:
            self.add_user(user_id)
            return self._history[user_id]
        else:
            return self._history[user_id]
