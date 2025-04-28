from typing import Dict, List

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
    ):
        self.tokenizer = rbpe_tokenizer.RBTokenizer(
            max_depth=8, tech_terms=tech_terms if tech_terms else []
        )

        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.problems = []

    def train_tokenizer(self, corpus: str):
        """Train RBPE tokenizer on full corpus"""
        if not isinstance(corpus, str):
            raise TypeError("Corpus must be a single string")

        self.tokenizer.train(corpus, vocab_size=32000)

    def create_embeddings(self, problems: List[Dict]):
        """Process LeetCode problems into FAISS index"""
        self.problems = problems
        embeddings = []

        for problem in self.problems:
            if "content" in problem and problem["content"]:
                text = f"{problem['title']} [SEP] {problem['content']}"

                chunks = self.tokenizer.chunk_with_overlap(
                    text, chunk_size=512, overlap=64
                )
                chunk_embeddings = []

                for chunk in chunks:
                    decoded_chunk = self.tokenizer.decode(chunk)
                    chunk_embedding = self.embedder.encode(
                        decoded_chunk, convert_to_tensor=False
                    )
                    chunk_embeddings.append(chunk_embedding)

                if chunk_embeddings:
                    embedding = np.mean(chunk_embeddings, axis=0)
                else:
                    embedding = self.embedder.encode(
                        problem["title"], convert_to_tensor=False
                    )
            else:
                tokens = self.tokenizer.encode_with_dropout(
                    problem["title"], dropout_prob=0.1
                )
                decoded_text = self.tokenizer.decode(tokens)
                embedding = self.embedder.encode(decoded_text, convert_to_tensor=False)

            embeddings.append(embedding)

        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)

        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """Search for similar LeetCode problems"""
        tokens = self.tokenizer.encode(question)
        tokenized_query = self.tokenizer.decode(tokens)

        query_embedding = self.embedder.encode(tokenized_query, convert_to_tensor=False)

        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )

        return [self.problems[i] for i in indices[0]]
