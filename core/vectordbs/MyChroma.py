from typing import List
import chromadb

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.schema.document import Document
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings as LangCEmbeddingFunc,
)
from core.utils.TextLlmUtils import TextLlmUtils


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pypdf import PdfReader

lc_embedding_function = LangCEmbeddingFunc(model_name="all-MiniLM-L6-v2")
chroma_embedding_function = SentenceTransformerEmbeddingFunction()


class LangchainChroma:
    def __init__(self) -> None:
        """initialize a langchain chroma vdb wrapper.

        Args:
            docs (List[Document]): _description_
        """
        self._vector_store = Chroma(
            collection_name="knpc",
            embedding_function=self.embedding_function,
            persist_directory="./chroma_db",
        )

    def add_documents(self, docs: List[Document]):
        ids = [str(e) for e in range(0, len(docs))]
        return self._vector_store.add_documents(documents=docs, ids=ids)

    def similarity_search(self, query: str, num_docs: int = 1):
        results = self._vector_store.similarity_search(query=query, k=num_docs)
        return results


class MyChroma:
    def __init__(self, embedding_func=chroma_embedding_function):
        self._client = chromadb.Client()
        self._embedding_function = embedding_func

    def read_pdf(self, filename):
        reader = PdfReader(filename)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        # Filter the empty strings
        pdf_texts = [text for text in pdf_texts if text]
        return pdf_texts

    def read_pdf_langchain_docs(self, path: str, chunk: int = 1000, overlap: int = 0):
        print(chunk)
        pdf_docs = TextLlmUtils.read_pdf(path=path)
        return pdf_docs

    def chunk_texts(self, texts):
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
        )
        character_split_texts = character_splitter.split_text("\n\n".join(texts))

        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=256
        )

        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        return token_split_texts

    def load_chroma_file(
        self, filename, embedding_function=None, collection_name: str = "default"
    ):
        texts = self.read_pdf(filename)
        chunks = self.chunk_texts(texts)
        embedding_func = (
            embedding_function
            if embedding_function is not None
            else self._embedding_function
        )
        chroma_client = self._client
        chroma_collection = chroma_client.create_collection(
            name=collection_name, embedding_function=embedding_func
        )

        ids = [str(i) for i in range(len(chunks))]

        chroma_collection.add(ids=ids, documents=chunks)

        return chroma_collection

    def load_chroma_docs(
        self,
        docs: List[Document],
        collection_name: str = "default",
        embedding_function=None,
    ):
        # pdf_docs = self.read_pdf_langchain_docs(path, chunk=chunk, overlap=overlap)
        embedding_func = (
            embedding_function
            if embedding_function is not None
            else self._embedding_function
        )
        chroma_collection = self._client.create_collection(
            name=collection_name, embedding_function=embedding_func
        )

        ids = [str(i) for i in range(len(docs))]
        metadata = [d.metadata for d in docs]
        content = [d.page_content for d in docs]

        chroma_collection.add(ids=ids, documents=content, metadatas=metadata)

        return chroma_collection
