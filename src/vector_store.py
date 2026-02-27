# -*- coding: utf-8 -*-
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from config import Config
from src.embeddings import get_embedding_model
import uuid
class VectorStore:
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.embedding_model = get_embedding_model()
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "knowledge base"})
        print(f"[OK] Vector database initialized: {self.collection_name}, docs: {self.collection.count()}")
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> List[str]:
        if not documents: return []
        if ids is None: ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedding_model.embed_texts(documents)
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas or [{} for _ in documents], ids=ids)
        print(f"[OK] Added {len(documents)} documents")
        return ids
    def add_test_case(self, test_case: Dict) -> str:
        content = f"ID: {test_case.get('test_id')}\nName: {test_case.get('test_name')}\nType: {test_case.get('test_type')}\nSteps: {test_case.get('steps')}"
        metadata = {"type": "test_case", "test_id": test_case.get("test_id", "")}
        return self.add_documents([content], [metadata])[0]
    def query(self, query_text: str, n_results: int = None, filter_dict: Optional[Dict] = None) -> Dict:
        n_results = n_results or Config.TOP_K
        query_embedding = self.embedding_model.embed_text(query_text)
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results, where=filter_dict)
    def search_similar_test_cases(self, requirement: str, top_k: int = 5, test_type: Optional[str] = None) -> List[Dict]:
        results = self.query(requirement, n_results=top_k)
        similar_cases = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                similar_cases.append({"content": doc, "metadata": results['metadatas'][0][i] if results['metadatas'] else {}, "distance": results['distances'][0][i] if results['distances'] else 0})
        return similar_cases
    def delete_all(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print("[OK] Cleared all documents")
        except Exception as e:
            print(f"[X] Delete failed: {e}")
    def get_stats(self) -> Dict:
        return {"total_documents": self.collection.count(), "collection_name": self.collection_name, "embedding_dimension": self.embedding_model.get_dimension()}
_vector_store = None
def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
