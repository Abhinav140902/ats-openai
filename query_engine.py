import json
import time
from typing import List, Dict, Optional, Generator, Tuple
from openai import OpenAI
import redis
from config import Config
from rich.console import Console
import hashlib

console = Console()

class QueryEngine:
    def __init__(self, hybrid_search):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.hybrid_search = hybrid_search
        self.model = Config.LLM_MODEL
        
        # Setup Redis cache
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL)
            self.redis_client.ping()
            self.use_cache = True
            console.print("[green]Redis cache connected[/green]")
        except:
            self.use_cache = False
            console.print("[yellow]Redis not available, caching disabled[/yellow]")
    
    def query(self, question: str, stream: bool = True) -> Generator[str, None, None]:
        """Answer questions using GPT with streaming."""
        total_start = time.time()
        
        # Check cache first
        cache_start = time.time()
        cache_key = f"query:{hashlib.md5(question.encode()).hexdigest()}"
        if self.use_cache:
            cached = self.redis_client.get(cache_key)
            if cached:
                cache_time = time.time() - cache_start
                console.print(f"[dim]📦 Cache hit! Retrieved in {cache_time*1000:.1f}ms[/dim]")
                yield cached.decode('utf-8')
                return
        cache_time = time.time() - cache_start
        
        # Search for relevant documents
        search_start = time.time()
        results = self.hybrid_search.search(question, k=Config.TOP_K_SEARCH)
        search_time = time.time() - search_start
        
        if not results:
            yield "No relevant information found in the resumes."
            return
        
        # Format context
        context_start = time.time()
        context = self._format_context(results)
        context_time = time.time() - context_start
        
        # Create prompt
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        # Calculate token count
        prompt_text = messages[0]["content"] + messages[1]["content"]
        prompt_tokens = len(prompt_text.split()) * 1.3  # Rough estimate
        
        console.print(f"\n[dim]🔍 Search time: {search_time*1000:.1f}ms | Found {len(results)} chunks[/dim]")
        console.print(f"[dim]📝 Context: ~{int(prompt_tokens)} tokens | {len(results)} documents[/dim]")
        console.print(f"[dim]🤖 Model: {self.model} | Streaming: {'Yes' if stream else 'No'}[/dim]\n")
        
        # Get response from GPT
        llm_start = time.time()
        first_token_time = None
        
        if stream and Config.ENABLE_STREAMING:
            full_response = ""
            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                stream=True
            )
            
            for i, chunk in enumerate(stream_response):
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time() - llm_start
                        console.print(f"[dim]⚡ First token: {first_token_time*1000:.1f}ms[/dim]\n")
                    
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            llm_time = time.time() - llm_start
            
            # Cache the complete response
            if self.use_cache and full_response:
                self.redis_client.setex(cache_key, Config.CACHE_TTL, full_response)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            llm_time = time.time() - llm_start
            answer = response.choices[0].message.content
            
            # Cache response
            if self.use_cache:
                self.redis_client.setex(cache_key, Config.CACHE_TTL, answer)
            
            yield answer
        
        # Print performance summary
        total_time = time.time() - total_start
        console.print("\n")
        console.print("[dim]⏱️  Performance Breakdown:[/dim]")
        console.print(f"[dim]   • Cache check: {cache_time*1000:.1f}ms[/dim]")
        console.print(f"[dim]   • Vector search: {search_time*1000:.1f}ms[/dim]") 
        console.print(f"[dim]   • Context prep: {context_time*1000:.1f}ms[/dim]")
        console.print(f"[dim]   • LLM response: {llm_time*1000:.1f}ms[/dim]")
        if first_token_time:
            console.print(f"[dim]   • Time to first token: {first_token_time*1000:.1f}ms[/dim]")
        console.print(f"[dim]   • Total time: {total_time*1000:.1f}ms[/dim]")
    
    def query_structured(self, question: str) -> Dict:
        """Get structured response for specific query types."""
        # Determine query type
        query_type = self._classify_query(question)
        
        # Search for relevant documents
        results = self.hybrid_search.search(question, k=Config.TOP_K_SEARCH)
        
        if not results:
            return {"error": "No relevant information found"}
        
        # Format context
        context = self._format_context(results)
        
        # Create structured prompt based on query type
        if query_type == "skill_search":
            response_format = {
                "type": "json_object",
                "schema": {
                    "candidates": [
                        {
                            "name": "string",
                            "filename": "string",
                            "has_skill": "boolean",
                            "evidence": "string"
                        }
                    ]
                }
            }
        elif query_type == "comparison":
            response_format = {
                "type": "json_object",
                "schema": {
                    "comparison": [
                        {
                            "candidate": "string",
                            "strengths": ["string"],
                            "experience_years": "number",
                            "key_skills": ["string"]
                        }
                    ],
                    "recommendation": "string"
                }
            }
        else:
            response_format = {"type": "json_object"}
        
        messages = [
            {"role": "system", "content": "You are an expert ATS system. Always respond with valid JSON."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a structured JSON response."}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            response_format=response_format
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"response": response.choices[0].message.content}
    
    def _format_context(self, results: List[Tuple[Dict, float]]) -> str:
        """Format search results as context for GPT."""
        context_parts = []
        
        for doc, score in results:
            context_parts.append(
                f"=== {doc['filename']} (Score: {score:.2f}) ===\n"
                f"Section: {doc.get('section', 'general')}\n"
                f"{doc['content']}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for GPT."""
        return """You are an expert ATS (Applicant Tracking System) assistant analyzing resumes.

Instructions:
1. Always identify candidates by their filename
2. For 'who' questions, list ALL matching candidates
3. Be specific and quote relevant information
4. If information is not found, say so clearly
5. For comparisons, create clear summaries

Keep responses concise and factual."""
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['who has', 'which candidate', 'list all']):
            return "skill_search"
        elif 'compare' in question_lower:
            return "comparison"
        elif 'rank' in question_lower or 'best' in question_lower:
            return "ranking"
        else:
            return "general"