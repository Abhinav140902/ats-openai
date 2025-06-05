import os
import time
import sys
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from config import Config
from resume_processor import ResumeProcessor
from embeddings_manager import EmbeddingsManager
from vector_store import VectorStore
from hybrid_search import HybridSearch
from query_engine import QueryEngine

console = Console()

def print_banner():
    banner = """
    🚀 AI-Powered ATS System v2.0
    ⚡ OpenAI GPT-4 • FAISS Vectors • Hybrid Search
    💨 10x Faster • Streaming Responses
    """
    console.print(Panel(banner, style="bold blue"))

def process_resumes():
    """Process and index all resumes."""
    processor = ResumeProcessor()
    vector_store = VectorStore()
    
    # Load resumes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading resumes...", total=None)
        resumes = processor.load_resumes()
        progress.update(task, completed=True)
    
    if not resumes:
        console.print("[red]No resumes found! Add PDF/DOCX files to data/resumes/[/red]")
        return None, None
    
    # Process resumes into chunks
    documents = []
    for filename, text in resumes:
        # Extract sections
        sections = processor.extract_sections(text)
        
        # Create chunks for each section
        chunk_id = 0
        for section_name, section_text in sections.items():
            if section_text:
                # Split long sections
                if len(section_text) > Config.CHUNK_SIZE:
                    chunks = split_text(section_text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                    for chunk in chunks:
                        documents.append({
                            'filename': filename,
                            'section': section_name,
                            'chunk_id': chunk_id,
                            'content': chunk,
                            'candidate': filename.replace('.pdf', '').replace('.docx', '')
                        })
                        chunk_id += 1
                else:
                    documents.append({
                        'filename': filename,
                        'section': section_name,
                        'chunk_id': chunk_id,
                        'content': section_text,
                        'candidate': filename.replace('.pdf', '').replace('.docx', '')
                    })
                    chunk_id += 1
    
    console.print(f"[green]Created {len(documents)} document chunks from {len(resumes)} resumes[/green]")
    
    # Create vector index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating vector index...", total=None)
        vector_store.create_index(documents)
        progress.update(task, completed=True)
    
    # Create hybrid search
    hybrid_search = HybridSearch(vector_store)
    hybrid_search.build_bm25_index(documents)
    
    return vector_store, hybrid_search

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def interactive_qa(query_engine):
    """Interactive Q&A session."""
    console.print("\n[bold green]🤖 AI-Powered Q&A Ready![/bold green]")
    console.print("Ask questions about your candidates. Type 'exit' to quit.\n")
    
    while True:
        try:
            question = console.input("[bold cyan]❓ Question:[/bold cyan] ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                break
            
            if not question:
                continue
            
            # Start timing
            start_time = time.time()
            
            console.print("\n[yellow]🤖 AI is thinking...[/yellow]")
            
            # Get streaming response
            console.print("[green]Answer:[/green] ", end="")
            for chunk in query_engine.query(question, stream=True):
                console.print(chunk, end="")
                sys.stdout.flush()
            
            console.print()  # New line after response
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def main():
    """Main application entry point."""
    print_banner()
    
    # Check API key
    if not Config.OPENAI_API_KEY:
        console.print("[red]❌ OPENAI_API_KEY not found![/red]")
        console.print("Add to .env file: OPENAI_API_KEY=sk-...")
        return
    
    try:
        # Process resumes
        console.print("\n[bold]📄 STEP 1: Processing Resumes[/bold]")
        vector_store, hybrid_search = process_resumes()
        
        if not vector_store:
            return
        
        # Initialize query engine
        console.print("\n[bold]🤖 STEP 2: Initializing AI Engine[/bold]")
        query_engine = QueryEngine(hybrid_search)
        console.print("[green]✅ AI Engine ready![/green]")
        
        # Start Q&A
        console.print("\n[bold]💬 STEP 3: Interactive Q&A[/bold]")
        interactive_qa(query_engine)
        
        console.print("\n[green]👋 Thanks for using ATS v2.0![/green]")
        
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()