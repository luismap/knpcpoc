from abc import ABC, abstractmethod
from typing import List


class QueryExpansion(ABC):
    @abstractmethod
    def query(question: str) -> List[str]:
        pass
