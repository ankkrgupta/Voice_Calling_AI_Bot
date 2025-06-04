import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

class RAGEngine:
    def __init__(self, 
                 openai_api_key: str,
                 pdf_path: str = "Zomato_Annual_Report_2023-24.pdf",
                 index_path: str = "Zomato_Annual_Report_2023-24.faiss",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "text-embedding-ada-002",
                 allow_dangerous: bool = True,
                 ):
        self.openai_api_key = openai_api_key
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
        if os.path.exists(self.index_path):
            self.db = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=allow_dangerous)
        else:
            self.db = None

    def build_index(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.db.save_local(self.index_path)

    def load_index(self):
        """Explicitly load an existing FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path!r}")
        self.db = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 5):
        if self.db is None:
            raise ValueError("Index not built. Call build_index() first.")
        docs = self.db.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)
        # return concatenated text of top-k
        return "\n---\n".join([doc.page_content for doc in docs])

# Integration in the LLM pipeline (llm_client.py)
class LLMClientWithRAG:
    def __init__(self, api_key: str, rag_engine: RAGEngine, model: str = "gpt-4o", temperature: float = 0.5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.history = []
        self.rag = rag_engine

    def reset(self):
        self.history = []

    async def stream_response(self, user_text: str):
        # Retrieve context
        context = self.rag.retrieve(user_text)
        # Prepend context to user message
        combined = f"Context:\n{context}\n\nQuestion: {user_text}"
        self.history.append({"role": "user", "content": combined})

        # Call the streaming chat completion
        def _sync():
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                stream=True,
            )
        import asyncio
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(None, _sync)

        full = ''
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full += delta
            yield delta
        self.history.append({"role": "assistant", "content": full})


'''Rag Code for HNSW Indexing'''

# import os
# import pickle
# import hnswlib
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings


# class RAGEngine:
#     """
#     RAG engine backed by hnswlib instead of FAISS.
#     - Splits a PDF into text chunks
#     - Embeds each chunk with OpenAIEmbeddings
#     - Builds (or loads) an hnswlib index over those embeddings
#     - At query time, embeds the query and does an approximate kNN lookup
#     - Returns the concatenated top-k chunk texts as context
#     """

#     def __init__(
#         self,
#         openai_api_key: str,
#         pdf_path: str = "Zomato_Annual_Report_2023-24.pdf",
#         index_dir: str = "zomato_hnsw_index",
#         chunk_size: int = 1000,
#         chunk_overlap: int = 200,
#         embedding_model: str = "text-embedding-ada-002",
#         ef_construction: int = 200,
#         M: int = 16,
#     ):
#         self.openai_api_key = openai_api_key
#         self.pdf_path = pdf_path
#         self.index_dir = index_dir  # directory where index + metadata will be saved
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         os.environ["OPENAI_API_KEY"] = openai_api_key

#         # Initialize embeddings client
#         self.embeddings = OpenAIEmbeddings(
#             model=embedding_model, openai_api_key=openai_api_key
#         )

#         # HNSW parameters
#         self.ef_construction = ef_construction
#         self.M = M

#         # Placeholders, to be set once index is built or loaded
#         self.index: hnswlib.Index = None
#         self.id2text: list[str] = []
#         self.dim: int = None

#         # Ensure index directory exists
#         os.makedirs(self.index_dir, exist_ok=True)

#         # If a saved index exists, load it
#         idx_path = os.path.join(self.index_dir, "hnsw_index.bin")
#         data_path = os.path.join(self.index_dir, "id2text.pkl")
#         meta_path = os.path.join(self.index_dir, "meta.pkl")

#         if os.path.exists(idx_path) and os.path.exists(data_path) and os.path.exists(meta_path):
#             self._load_index(idx_path, data_path, meta_path)
#         else:
#             self.index = None  # will trigger build on first retrieve()

#     def build_index(self):
#         # 1) Load PDF and split into chunks
#         loader = PyPDFLoader(self.pdf_path)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
#         )
#         chunks = splitter.split_documents(documents)

#         # 2) Compute embeddings for each chunk
#         texts = [chunk.page_content for chunk in chunks]
#         # Embed in batches to avoid rate-limits
#         all_embeds = []
#         batch_size = 32
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i : i + batch_size]
#             batch_embeds = self.embeddings.embed_documents(batch_texts)
#             # openai embed_documents returns List[List[float]]
#             all_embeds.extend(batch_embeds)
#         # Now all_embeds is a list of length len(texts), each a vector of size dim
#         import numpy as np
#         embeds_np = np.array(all_embeds, dtype="float32")
#         self.dim = embeds_np.shape[1]

#         # 3) Create HNSW index
#         num_elements = embeds_np.shape[0]
#         self.index = hnswlib.Index(space="cosine", dim=self.dim)

#         # Initialize index; max_elements must be ≥ num_elements
#         self.index.init_index(
#             max_elements=num_elements,
#             ef_construction=self.ef_construction,
#             M=self.M,
#         )

#         # Add vectors with integer IDs 0..num_elements-1
#         ids = list(range(num_elements))
#         self.index.add_items(embeds_np, ids)
#         # Set ef (controls recall vs. speed at query time). Higher ef → better recall, slower.
#         self.index.set_ef(50)

#         # 4) Save mapping from ID → chunk text
#         self.id2text = texts

#         # 5) Persist index + metadata to disk
#         idx_path = os.path.join(self.index_dir, "hnsw_index.bin")
#         data_path = os.path.join(self.index_dir, "id2text.pkl")
#         meta_path = os.path.join(self.index_dir, "meta.pkl")

#         # hnswlib has its own save/load API
#         self.index.save_index(idx_path)

#         # Save id2text list
#         with open(data_path, "wb") as f:
#             pickle.dump(self.id2text, f)

#         # Save metadata (e.g. dimension)
#         with open(meta_path, "wb") as f:
#             pickle.dump({"dim": self.dim}, f)

#     def _load_index(self, idx_path: str, data_path: str, meta_path: str):
#         # Load metadata first to get dim
#         import pickle
#         with open(meta_path, "rb") as f:
#             meta = pickle.load(f)
#         self.dim = meta["dim"]

#         # Recreate index and load
#         self.index = hnswlib.Index(space="cosine", dim=self.dim)
#         self.index.load_index(idx_path)
#         # After loading, you must set ef (query-time parameter)
#         self.index.set_ef(50)

#         # Load id→text mapping
#         with open(data_path, "rb") as f:
#             self.id2text = pickle.load(f)

#     def retrieve(self, query: str, k: int = 5):
#         """
#         Return the concatenated top-k chunk texts for the given query.
#         Builds the index if it doesn’t exist yet.
#         """
#         # If no index on startup, build it
#         if self.index is None:
#             self.build_index()

#         # 1) Embed the query
#         q_embed = self.embeddings.embed_query(query)
#         import numpy as np
#         q_embed_np = np.array(q_embed, dtype="float32").reshape(1, -1)

#         # 2) Query hnswlib for top-k
#         labels, distances = self.index.knn_query(q_embed_np, k=k)
#         # labels is shape (1, k), distances same

#         # 3) Gather texts
#         results = []
#         for idx in labels[0]:
#             # Some robust check (in case k > actual elements)
#             if 0 <= idx < len(self.id2text):
#                 results.append(self.id2text[idx])

#         # 4) Return concatenated
#         return "\n---\n".join(results)
