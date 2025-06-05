# 🚀 AI-Powered ATS (Applicant Tracking System)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://openai.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-fast, AI-powered resume screening system that uses OpenAI's GPT-4o-mini, FAISS vector search, and hybrid search algorithms to intelligently match candidates to job requirements.

## ✨ Features

- **🎯 Smart Resume Parsing**: Extracts text from PDF and DOCX files
- **🔍 Hybrid Search**: Combines semantic (FAISS) and keyword (BM25) search
- **💬 Natural Language Q&A**: Ask questions about candidates in plain English
- **⚡ Streaming Responses**: See results as they generate (0.6s to first token)
- **📊 Performance Metrics**: Detailed breakdown of query performance
- **💾 Intelligent Caching**: Redis cache for instant repeated queries
- **🐳 Docker Ready**: Full containerization with docker-compose

## 📈 Performance

| Metric | Performance | vs Claude |
|--------|------------|-----------|
| First Token | ~600ms | 5x faster |
| Total Response | ~3.9s | Similar |
| Search Time | ~970ms | N/A |
| Cost per Query | $0.0008 | 75% cheaper |

## 🛠️ Tech Stack

- **AI/ML**: OpenAI GPT-4o-mini, text-embedding-3-small, FAISS
- **Backend**: Python 3.10, asyncio
- **Search**: Hybrid (FAISS + BM25)
- **Cache**: Redis
- **Infrastructure**: Docker, docker-compose
- **UI**: Rich terminal interface

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-powered-ats.git
cd ai-powered-ats
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. **Add resume files**
```bash
mkdir -p data/resumes
# Copy your PDF/DOCX resumes to data/resumes/
```

4. **Run with Docker**
```bash
docker-compose up -d
docker exec -it ai-powered-ats-ats-app-1 python app.py
```

## 💡 Usage

Once running, you can ask questions like:

```
❓ Who has Python experience?
❓ Which candidates have internship experience?
❓ Compare the technical skills of all candidates
❓ Who is best suited for a backend developer role?
❓ List all candidates with more than 2 years experience
```

### Example Output

```
🔍 Search time: 972.3ms | Found 5 chunks
📝 Context: ~721 tokens | 5 documents
🤖 Model: gpt-4o-mini | Streaming: Yes

⚡ First token: 609.6ms

Candidates with Python experience:
1. **Abhinav_s_Resume.pdf**: Python, React, Node.js...
2. **yasin.pdf**: Developed project in Python...

⏱️ Performance Breakdown:
   • Vector search: 972.3ms
   • LLM response: 2959.5ms
   • Total time: 3936.7ms
```

## 📁 Project Structure

```
ai-powered-ats/
├── app.py                 # Main application
├── config.py              # Configuration settings
├── embeddings_manager.py  # OpenAI embeddings handler
├── vector_store.py        # FAISS vector database
├── hybrid_search.py       # BM25 + vector search
├── query_engine.py        # GPT-4o-mini query handler
├── resume_processor.py    # PDF/DOCX text extraction
├── docker-compose.yml     # Docker orchestration
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── data/
    ├── resumes/         # Place resume files here
    ├── faiss_index/     # Vector index storage
    └── cache/           # Embeddings cache
```

## ⚙️ Configuration

Key settings in `config.py`:

```python
LLM_MODEL = "gpt-4o-mini"  # Or "gpt-4-turbo" for better quality
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
VECTOR_SEARCH_WEIGHT = 0.7
KEYWORD_SEARCH_WEIGHT = 0.3
```

## 🔧 Advanced Usage

### Using Different Models

```python
# In config.py
LLM_MODEL = "gpt-4-turbo"  # Better quality, slower
LLM_MODEL = "gpt-3.5-turbo"  # Older, slightly cheaper
```

### Adjusting Search Weights

```python
# More semantic search
VECTOR_SEARCH_WEIGHT = 0.9
KEYWORD_SEARCH_WEIGHT = 0.1
```

### Custom Chunking

```python
# Smaller chunks for more precise search
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
```

## 📊 Cost Analysis

- **Embeddings**: ~$0.00002 per resume
- **Queries**: ~$0.0008 per question
- **Monthly estimate**: $8 for 10,000 queries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embeddings
- Facebook AI for FAISS
- The Python community for excellent libraries

## 📞 Support

- Create an [Issue](https://github.com/yourusername/ai-powered-ats/issues)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

Built with ❤️ by [Your Name]