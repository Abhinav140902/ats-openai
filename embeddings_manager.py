import os
import json
import hashlib
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
import tiktoken
from diskcache import Cache
from config import Config
from rich.console import Console

console = Console()

class EmbeddingsManager:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.EMBEDDING_MODEL
        self.cache = Cache(os.path.join(Config.CACHE_DIR, 'embeddings'))
        # Use cl100k_base encoding which is compatible with embedding models
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    def get_embedding(self, text: str, cache_key: Optional[str] = None) -> List[float]:
        """Get embedding for text with caching."""
        # Use provided cache key or generate from text
        if not cache_key:
            cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Truncate if too long
        tokens = self.encoder.encode(text)
        if len(tokens) > 8191:  # Max tokens for embedding model
            text = self.encoder.decode(tokens[:8191])
        
        # Get embedding from OpenAI
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache permanently
            self.cache.set(cache_key, embedding, expire=None)
            
            return embedding
        except Exception as e:
            console.print(f"[red]Embedding error: {e}[/red]")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached = self.cache.get(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch process uncached texts
        if uncached_texts:
            console.print(f"[yellow]Getting embeddings for {len(uncached_texts)} texts...[/yellow]")
            
            # Process in batches of 100
            batch_size = 100
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    for j, embedding_data in enumerate(response.data):
                        idx = uncached_indices[i + j]
                        embedding = embedding_data.embedding
                        embeddings[idx] = embedding
                        
                        # Cache it
                        cache_key = hashlib.md5(texts[idx].encode()).hexdigest()
                        self.cache.set(cache_key, embedding, expire=None)
                        
                except Exception as e:
                    console.print(f"[red]Batch embedding error: {e}[/red]")
                    # Fill with zero vectors
                    for j in range(len(batch)):
                        idx = uncached_indices[i + j]
                        embeddings[idx] = [0.0] * 1536
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self.cache.clear()
        console.print("[green]Embeddings cache cleared[/green]")