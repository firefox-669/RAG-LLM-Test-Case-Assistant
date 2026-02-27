# -*- coding: utf-8 -*-
"""
RAG Chain - Simplified
"""

from typing import List, Dict
from config import Config
from src.vector_store import get_vector_store
from src.llm_handler import get_llm_handler


class RAGChain:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm_handler = get_llm_handler()
        print("[OK] RAG chain initialized")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or Config.TOP_K
        results = self.vector_store.search_similar_test_cases(query, top_k=top_k)
        return results

    def generate_with_context(self, query: str, context_docs: List[Dict], prompt_template: str) -> str:
        context_text = self._format_context(context_docs)
        prompt = prompt_template.format(requirement=query, context=context_text, query=query, test_cases=query)
        response = self.llm_handler.generate_with_context(Config.SYSTEM_PROMPT, prompt)
        return response

    def run(self, query: str, prompt_template: str, top_k: int = None) -> Dict:
        retrieved_docs = self.retrieve(query, top_k)
        generated_text = self.generate_with_context(query, retrieved_docs, prompt_template)
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_text": generated_text,
            "num_docs_retrieved": len(retrieved_docs)
        }

    @staticmethod
    def _format_context(docs: List[Dict]) -> str:
        if not docs:
            return "No relevant reference cases available."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            similarity = 1 - doc.get('distance', 0)
            context_parts.append(f"Reference case {i} (similarity: {similarity:.2f}):\n{doc['content']}\n")

        return "\n".join(context_parts)


_rag_chain = None

def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
