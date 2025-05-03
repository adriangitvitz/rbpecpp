import os
from typing import Dict, List, Optional

import faiss
import numpy as np
import rbpe_tokenizer
from sentence_transformers import SentenceTransformer


class VectorSearchSystem:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vocab_size: int = 32000,
        tech_terms: List[str] = None,
        persist_dir: str = "./vector_store",
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.tokenizer = self._load_tokenizer(tech_terms)
        self.embedder = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.problems: List[Dict] = []
        self.token_cache: Dict[str, List[int]] = {}
        self.id_map = []

    def _load_tokenizer(
        self, tech_terms: List[str] = None
    ) -> rbpe_tokenizer.RBTokenizer:
        tokenizer_path = os.path.join(self.persist_dir, "tokenizer_state.bin")
        if os.path.exists(tokenizer_path):
            tokenizer = rbpe_tokenizer.RBTokenizer(max_depth=8)
            tokenizer.load(tokenizer_path)
            return tokenizer
        return rbpe_tokenizer.RBTokenizer(
            max_depth=8, tech_terms=tech_terms if tech_terms else []
        )

    def train_tokenizer(self, corpus: str):
        """Train and persist RBPE tokenizer state"""
        self.tokenizer.train(corpus, 32000)
        self.tokenizer.save(os.path.join(self.persist_dir, "tokenizer_state.bin"))

    def create_embeddings(self, problems: List[Dict], batch_size: int = 64):
        self.problems = problems
        embeddings = []
        token_store = []

        valid_indexes = []
        for idx, p in enumerate(self.problems):
            try:
                problem_id = str(p["id"])
                int(problem_id)
                valid_indexes.append(idx)
                self.id_map.append(problem_id)
            except (ValueError, KeyError, TypeError):
                print(f"Skipping invalid ID: {p.get('id', 'MISSING')}")
                continue

        for i in range(0, len(problems), batch_size):
            batch_indices = valid_indexes[i : i + batch_size]
            batch = [problems[idx] for idx in batch_indices]
            batch_tokens = []

            for problem in batch:
                if "content" in problem and problem["content"]:
                    text = f"{problem['title']} [SEP] {problem['content']}"
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = self.tokenizer.encode_with_dropout(
                        problem["title"], dropout_prob=0.1
                    )

                self.token_cache[problem["id"]] = tokens
                batch_tokens.append(tokens)

            batch_texts = [self.tokenizer.decode(tokens) for tokens in batch_tokens]
            batch_embeddings = self.embedder.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )

            embeddings.extend(batch_embeddings)
            token_store.extend(batch_tokens)

        dim = embeddings[0].shape[0]

        embeddings_np = np.array(embeddings).astype("float32")
        np.save(os.path.join(self.persist_dir, "embeddings.npy"), embeddings_np)
        quantizer = faiss.IndexFlatIP(dim)
        nlist = 100
        self.index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(embeddings_np)
        ids_np = np.array(valid_indexes, dtype=np.int64)
        self.index.add_with_ids(embeddings_np, ids_np)

        faiss.write_index(
            self.index, os.path.join(self.persist_dir, "leetcode_index.faiss")
        )

    def load_embeddings(self):
        """Load persisted embeddings and index"""
        embeddings_path = os.path.join(self.persist_dir, "embeddings.npy")
        index_path = os.path.join(self.persist_dir, "leetcode_index.faiss")

        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            self.index = faiss.read_index(index_path)
            self.id_map = [
                p["id"] for p in self.problems if p["id"] in self.token_cache
            ]

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        query_tokens = self.tokenizer.encode(question)

        valid_cache = {
            p["id"]: self.token_cache[p["id"]]
            for p in self.problems
            if p["id"] in self.token_cache
        }

        token_overlap_scores = [
            (pid, len(set(query_tokens) & set(tokens)))
            for pid, tokens in valid_cache.items()
        ]
        token_overlap_scores.sort(key=lambda x: x[1], reverse=True)

        candidate_ids = [str(pid) for pid, _ in token_overlap_scores[: top_k * 3]]

        valid_indices = [
            idx
            for idx, mapped_id in enumerate(self.id_map)
            if mapped_id in set(candidate_ids)
        ]

        if not valid_indices:
            print("Nothing found")
            return []

        indices_np = np.array(valid_indices, dtype=np.int64)

        sel = faiss.IDSelectorBatch(indices_np.size, faiss.swig_ptr(indices_np))

        query_embedding = self.embedder.encode(
            self.tokenizer.decode(query_tokens),
            convert_to_tensor=False,
            normalize_embeddings=True,
        ).astype("float32")

        params = faiss.SearchParametersIVF(
            nprobe=10,
            sel=sel,
        )

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k, params=params
        )

        valid_results = [i for i in indices[0] if i != -1]
        return [self.problems[i] for i in valid_results]
