from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from os import walk
import re
import numpy as np
from tqdm import tqdm


class TextLlmUtils:
    """Utils class for interacting with different methods needed by the llm
    to handle text data
    """

    def project_embeddings(embeddings, umap_transform):
        umap_embeddings = np.empty((len(embeddings), 2))
        for i, embedding in enumerate(tqdm(embeddings)):
            umap_embeddings[i] = umap_transform.transform([embedding])
        return umap_embeddings

    def find_pdf_files(path: str) -> List[str]:
        pdf_files = []
        # List all PDF files in the folder
        for root, subdirs, files in walk(path):
            pdf_files.extend(
                [root + "/" + file for file in files if file.endswith(".pdf")]
            )
        return pdf_files

    def read_pdf(path: str, chunk=1000, overlap=0) -> List[Document]:
        """given a paht, find all pdf files and.
            1. read and parse into docs
            2. clean empty docs
            3. clean extra characters like spaces, ect
            4. chunk it
        Args:
            path (str): _description_
            chunk (int, optional): _description_. Defaults to 1000.
            overlap (int, optional): _description_. Defaults to 0.

        Returns:
            List[Document]: _description_
        """
        pdf_files = TextLlmUtils.find_pdf_files(path)
        print(f"found {pdf_files}")
        documents = []
        for file in pdf_files:
            print(f"loading doc {file}")
            documents.extend(PyPDFLoader(file).load())

        print("cleaning pdf docs")
        rm_empty_docs = list(filter(lambda doc: doc.page_content != "", documents))
        clean_docs = [TextLlmUtils.clean_document(doc) for doc in rm_empty_docs]
        print("chunking pdf docs")
        chunk_docs = TextLlmUtils.split(
            clean_docs, chunk_size=chunk, chunk_overlap=overlap
        )
        return chunk_docs

    def clean_document(doc: Document) -> Document:
        cleaned = re.sub("\n{2,}|\t+|\s{2,}", " ", doc.page_content).strip()
        new_doc = Document(page_content=cleaned, metadata=doc.metadata, type=doc.type)
        return new_doc

    def loader(path: str) -> List[Document]:
        """given a file path, retrieve a list of (langchain) Documents
        Args:
            path (str): local path

        Returns:
            List[Document]: a list of documents
        """
        loader = TextLoader(path)
        docs = loader.load()
        return docs

    def webloader(links: List[str]) -> List[Document]:
        """given a list of web links, parse and create a documents
        list.
        Args:
            links (List[str]): list of webpages

        Returns:
            List[Document]: a list of documents
        """
        # TODO clean newlines and trim content
        loader = WebBaseLoader(links)
        docs = loader.load()
        cleaned_docs = [TextLlmUtils.clean_document(doc) for doc in docs]
        return cleaned_docs

    def split(docs: List[Document], chunk_size=600, chunk_overlap=10) -> List[Document]:
        """given a list of documents, create splits for those documents base
        om some criteria like chunk_size and chunk_overlap.
        When finding the right balance, tweak the chunksize fo find which is the
        best for your usecase

        Args:
            docs (List[Document]): list of documents
            chunk_size (int, optional): chunk size.
            chunk_overlap (int, optional): overlapping chunks. Defaults to 0.

        Returns:
            List[Document]: a list of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)
        return splits

    def hugginface_embeddings(
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs: dict = {"device": "cpu"},
        encode_kwargs: dict = {"normalize_embeddings": False},
    ) -> HuggingFaceEmbeddings:
        """create a huggingfaceembedding model to be use as an embedding algorithm

        Args:
            model_name (str, optional): embeddings model to use. Defaults to "sentence-transformers/all-mpnet-base-v2".
            model_kwargs (_type_, optional): model kwargs. Defaults to {'device': 'cpu'}.
            encode_kwargs (_type_, optional): encode kwargs. Defaults to {'normalize_embeddings': False}.

        Returns:
            HuggingFaceEmbeddings: an embedding from langchain lib
        """
        hfe = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return hfe

    def from_doc_to_text(docs: List[Document]) -> List[str]:
        return [doc.page_content for doc in docs]

    def format_docs(docs: List[Document]) -> str:
        """Given a list of documents. Returns a string with the documents
        concatenated. This can be use to pass it as a context to an LLM for
        RAG aware information

        Args:
            docs (List[Document]): the document list

        Returns:
            str: documents content concatenated
        """
        return "\n\n".join(doc.page_content for doc in docs)
