import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from rich.console import Console
from config import Config
from embeddings_manager import EmbeddingsManager

console = Console()

class VectorStore:
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.index = None
        self.metadata = []
        self.dimension = 1536  # OpenAI embedding dimension
        self.index_path = Config.FAISS_INDEX_PATH
        os.makedirs(self.index_path, exist_ok=True)
        
    def create_index(self, documents: List[Dict[str, str]]):
        """Create FAISS index from documents."""
        console.print(f"[green]Creating vector index for {len(documents)} documents...[/green]")
        
        # Get embeddings for all documents
        texts = [doc['content'] for doc in documents]
        embeddings = self.embeddings_manager.get_embeddings_batch(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # For larger datasets, use IVF for faster search
        if len(documents) > 1000:
            nlist = min(100, len(documents) // 10)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings_array)
        
        # Add vectors to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata = documents
        
        # Save index
        self._save_index()
        
        console.print(f"[green]✅ Vector index created with {len(documents)} documents[/green]")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents."""
        if self.index is None:
            self._load_index()
            if self.index is None:
                return []
        
        # Get query embedding
        query_embedding = self.embeddings_manager.get_embedding(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distance)
                results.append((self.metadata[idx], similarity))
        
        return results
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add new documents to existing index."""
        if self.index is None:
            self.create_index(documents)
            return
        
        # Get embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.embeddings_manager.get_embeddings_batch(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        self.metadata.extend(documents)
        
        # Save updated index
        self._save_index()
        
        console.print(f"[green]Added {len(documents)} documents to index[/green]")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            
            # Save metadata
            with open(os.path.join(self.index_path, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.metadata, f)
                
        except Exception as e:
            console.print(f"[red]Error saving index: {e}[/red]")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                console.print(f"[green]Loaded index with {self.index.ntotal} vectors[/green]")
            else:
                console.print("[yellow]No existing index found[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error loading index: {e}[/red]")
    
    def clear(self):
        """Clear the index."""
        self.index = None
        self.metadata = []
        
        # Remove saved files
        for file in ["index.faiss", "metadata.pkl"]:
            path = os.path.join(self.index_path, file)
            if os.path.exists(path):
                os.remove(path)