import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"  # Fast and cheap
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 1000
    
    # Search Settings
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    VECTOR_SEARCH_WEIGHT = 0.7
    KEYWORD_SEARCH_WEIGHT = 0.3
    TOP_K_SEARCH = 5
    
    # Cache Settings
    REDIS_URL = "redis://redis:6379"
    CACHE_TTL = 86400  # 24 hours
    EMBEDDING_CACHE_TTL = None  # Permanent
    
    # File Paths
    RESUME_DIR = "data/resumes"
    FAISS_INDEX_PATH = "data/faiss_index"
    CACHE_DIR = "data/cache"
    
    # Performance
    ENABLE_STREAMING = True
    BATCH_SIZE = 10
    MAX_WORKERS = 4