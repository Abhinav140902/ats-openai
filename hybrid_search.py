from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from config import Config
from rich.console import Console

console = Console()

class HybridSearch:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        
    def build_bm25_index(self, documents: List[Dict[str, str]]):
        """Build BM25 index for keyword search."""
        console.print("[yellow]Building BM25 index...[/yellow]")
        
        self.documents = documents
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization - can be improved with NLTK
            tokens = doc['content'].lower().split()
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        console.print("[green]BM25 index ready[/green]")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform hybrid search combining vector and keyword search."""
        results = {}
        
        # Vector search
        vector_results = self.vector_store.search(query, k=k*2)
        for doc, score in vector_results:
            doc_id = f"{doc['filename']}_{doc.get('chunk_id', 0)}"
            results[doc_id] = {
                'doc': doc,
                'vector_score': score,
                'keyword_score': 0.0
            }
        
        # BM25 keyword search
        if self.bm25:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top k*2 results from BM25
            top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
            
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_id = f"{doc['filename']}_{doc.get('chunk_id', 0)}"
                    
                    # Normalize BM25 score to 0-1
                    max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
                    normalized_score = bm25_scores[idx] / max_score
                    
                    if doc_id in results:
                        results[doc_id]['keyword_score'] = normalized_score
                    else:
                        results[doc_id] = {
                            'doc': doc,
                            'vector_score': 0.0,
                            'keyword_score': normalized_score
                        }
        
        # Combine scores
        final_results = []
        for doc_id, scores in results.items():
            combined_score = (
                Config.VECTOR_SEARCH_WEIGHT * scores['vector_score'] +
                Config.KEYWORD_SEARCH_WEIGHT * scores['keyword_score']
            )
            final_results.append((scores['doc'], combined_score))
        
        # Sort by combined score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:k]
    
    def search_with_filter(self, query: str, filters: Dict, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search with metadata filters."""
        all_results = self.search(query, k=k*3)
        
        filtered_results = []
        for doc, score in all_results:
            # Apply filters
            match = True
            for key, value in filters.items():
                if key in doc and doc[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append((doc, score))
        
        return filtered_results[:k]