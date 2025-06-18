import os
import pickle
import hnswlib
import numpy as np
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

class RAGEngine:
    """
    RAG engine backed by hnswlib with a lightweight pre-filter:
    - Splits a PDF into text chunks
    - Embeds each chunk with OpenAIEmbeddings (full index) and a local MiniLM (fast index)
    - At query time, runs a fast relevance check via MiniLM->HNSW; if above threshold,
      falls back to the full OpenAI embed + HNSW lookup.
    - Returns the concatenated top-k chunk texts as context
    """

    def __init__(
        self,
        openai_api_key: str,
        pdf_path: str = "Zomato_Annual_Report_2023-24.pdf",
        index_path: str = "zomato_hnsw_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-ada-002",
        ef_construction: int = 200,
        M: int = 16,
        fast_threshold: float = 0.60,
    ):
        self.openai_api_key = openai_api_key
        self.pdf_path = pdf_path
        self.index_dir = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Fast pre-filter: MiniLM embedder + HNSW
        self.fast_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.fast_threshold = fast_threshold
        self.fast_index: hnswlib.Index = None
        self.fast_dim: int = None

        # Full RAG: OpenAI embeddings + full-dim HNSW
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model, openai_api_key=openai_api_key
        )
        self.ef_construction = ef_construction
        self.M = M
        self.index: hnswlib.Index = None
        self.id2text: list[str] = []
        self.dim: int = None

        os.makedirs(self.index_dir, exist_ok=True)
        # memory subdir
        self.memory_dir = os.path.join(self.index_dir, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memories_file = os.path.join(self.memory_dir, "user_memories.pkl")
        self.user_memories = pickle.load(open(self.memories_file, "rb")) if os.path.exists(self.memories_file) else {}

        # Load full index if exists
        idx_path = os.path.join(self.index_dir, "hnsw_index.bin")
        data_path = os.path.join(self.index_dir, "id2text.pkl")
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        if os.path.exists(idx_path) and os.path.exists(data_path) and os.path.exists(meta_path):
            self.load_index(idx_path, data_path, meta_path)
        # Load fast index if exists
        fast_idx = os.path.join(self.index_dir, "fast_index.bin")
        fast_meta = os.path.join(self.index_dir, "fast_meta.pkl")
        if os.path.exists(fast_idx) and os.path.exists(fast_meta):
            self.fast_dim = pickle.load(open(fast_meta, "rb"))["dim"]
            self.fast_index = hnswlib.Index(space="cosine", dim=self.fast_dim)
            self.fast_index.load_index(fast_idx)
            self.fast_index.set_ef(20)


    def build_index(self):
        # 1) Load PDF and split
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]

        # 2) Full-embeds
        all_full = []
        for i in range(0, len(texts), 32):
            all_full.extend(self.embeddings.embed_documents(texts[i:i+32]))
        full_np = np.array(all_full, dtype="float32")
        self.dim = full_np.shape[1]

        # 3) Build full HNSW
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=len(full_np), ef_construction=self.ef_construction, M=self.M)
        self.index.add_items(full_np, np.arange(len(full_np)))
        self.index.set_ef(50)

        # 4) Persist full
        pickle.dump(texts, open(os.path.join(self.index_dir, "id2text.pkl"), "wb"))
        pickle.dump({"dim": self.dim}, open(os.path.join(self.index_dir, "meta.pkl"), "wb"))
        self.index.save_index(os.path.join(self.index_dir, "hnsw_index.bin"))
        self.id2text = texts

        # 5) Build fast index on MiniLM embeddings
        fast_embeds = self.fast_embedder.encode(texts, convert_to_numpy=True)
        self.fast_dim = fast_embeds.shape[1]
        self.fast_index = hnswlib.Index(space="cosine", dim=self.fast_dim)
        self.fast_index.init_index(max_elements=len(fast_embeds), ef_construction=self.ef_construction, M=self.M)
        self.fast_index.add_items(fast_embeds, np.arange(len(fast_embeds)))
        self.fast_index.set_ef(20)
        # Persist fast
        self.fast_index.save_index(os.path.join(self.index_dir, "fast_index.bin"))
        pickle.dump({"dim": self.fast_dim}, open(os.path.join(self.index_dir, "fast_meta.pkl"), "wb"))

    def load_index(self, idx_path: str, data_path: str, meta_path: str):
        self.dim = pickle.load(open(meta_path, "rb"))["dim"]
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(idx_path)
        self.index.set_ef(20)
        self.id2text = pickle.load(open(data_path, "rb"))

    def _fast_relevance(self, query: str) -> float:
        # Ensure fast index is built
        if self.fast_index is None:
            self.build_index()
        # local embed
        vec = self.fast_embedder.encode([query], convert_to_numpy=True)
        # knn
        _, d = self.fast_index.knn_query(vec, k=1)
        # cosine similarity = 1 - cosine distance
        sim = 1.0 - float(d[0][0])
        return sim

    def retrieve(self, query: str, k: int = 5) -> str:
        # Bootstrap indices if needed
        if self.index is None or self.fast_index is None:
            self.build_index()
        # fast prefilter
        if self._fast_relevance(query) < self.fast_threshold:
            return ""
        # full retrieval
        full_vec = self.embeddings.embed_query(query)
        q_np = np.array(full_vec, dtype="float32").reshape(1, -1)
        labels, _ = self.index.knn_query(q_np, k=k)
        return "\n---\n".join(self.id2text[i] for i in labels[0] if i < len(self.id2text))

    def build_memory_index(self, summaries: list[str]):
        embeds = self.embeddings.embed_documents(summaries)
        arr = np.array(embeds, dtype="float32")
        dim = arr.shape[1]
        self.mem_index = hnswlib.Index(space="cosine", dim=dim)
        self.mem_index.init_index(max_elements=max(50+len(summaries),50*len(summaries)), ef_construction=self.ef_construction, M=self.M)
        self.mem_index.add_items(arr, list(range(len(summaries))))
        self.mem_index.set_ef(50)
        pickle.dump(summaries, open(os.path.join(self.memory_dir, "id2summary.pkl"), "wb"))
        pickle.dump({"dim": dim}, open(os.path.join(self.memory_dir, "meta_memory.pkl"), "wb"))
        self.mem_summaries = summaries
        self.mem_index.save_index(os.path.join(self.memory_dir, "hnsw_memory.bin"))

    def load_memory_index(self, idx_path, data_path, meta_path):
        dim = pickle.load(open(meta_path, "rb"))["dim"]
        self.mem_index = hnswlib.Index(space="cosine", dim=dim)
        self.mem_index.load_index(idx_path)
        self.mem_index.set_ef(50)
        self.mem_summaries = pickle.load(open(data_path, "rb"))

    def add_memory(self, key: str, summary: str):
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = f"[{ts}] {summary}"
        bucket = self.user_memories.setdefault(key, [])
        bucket.append(entry)
        if len(bucket) > 5:
            self.user_memories[key] = bucket[-5:]
        pickle.dump(self.user_memories, open(self.memories_file, "wb"))

    def retrieve_memory(self, key: str, k: int = 3) -> str:
        bucket = self.user_memories.get(key, [])
        recent = bucket[-k:]
        return "\n".join(recent)
