#!/usr/bin/env python3
"""
GitHub MCP Tools Chatbot
RAG-based chatbot for querying GitHub MCP tools using Weaviate and Gemini.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

# Import services from data processor
from data_processor import EmbeddingService, WeaviateService
from weaviate.classes.query import MetadataQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class WeaviateQueryService(WeaviateService):
    """Extends WeaviateService for query-only operations"""
    
    def check_data_exists(self) -> bool:
        """Check if the collection exists and has data"""
        try:
            if not self.client.collections.exists(self.collection_name):
                return False
            
            collection = self.client.collections.get(self.collection_name)
            # Try to get at least one object to verify data exists
            response = collection.query.fetch_objects(limit=1)
            return len(response.objects) > 0
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check collection status: {e}[/yellow]")
            return False
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            collection = self.client.collections.get(self.collection_name)
            
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=MetadataQuery(certainty=True)
            )
            
            # Convert response to the expected format
            results = []
            for obj in response.objects:
                result = {
                    "section": obj.properties.get("section"),
                    "toolName": obj.properties.get("toolName"),
                    "description": obj.properties.get("description"),
                    "parameters": obj.properties.get("parameters"),
                    "fullText": obj.properties.get("fullText"),
                    "chunkId": obj.properties.get("chunkId"),
                    "_additional": {
                        "certainty": obj.metadata.certainty if obj.metadata else None
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            console.print(f"[red]✗ Search failed: {e}[/red]")
            raise

class GeminiService:
    """Handles Gemini LLM interactions"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._configure()
    
    def _configure(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            console.print(f"[green]✓ Gemini {self.model_name} configured successfully[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to configure Gemini: {e}[/red]")
            raise
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context"""
        
        # Build context from retrieved chunks
        context = "AVAILABLE GITHUB MCP TOOLS:\n\n"
        for i, chunk in enumerate(context_chunks, 1):
            context += f"Tool {i}:\n"
            context += f"Section: {chunk['section']}\n"
            context += f"Name: {chunk['toolName']}\n"
            context += f"Description: {chunk['description']}\n"
            
            # Parse parameters if they exist
            try:
                params = json.loads(chunk['parameters'])
                if params:
                    context += "Parameters:\n"
                    for param_name, param_desc in params.items():
                        context += f"  - {param_name}: {param_desc}\n"
            except:
                pass
            
            context += "\n" + "-" * 50 + "\n\n"
        
        # System prompt for grounding
        system_prompt = """You are a helpful assistant for GitHub MCP (Model Context Protocol) tools. 

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided tool information
2. If the question cannot be answered from the provided tools, say so explicitly
3. Be specific about tool names, sections, and parameters
4. Provide code examples when relevant
5. If multiple tools could help, mention all relevant ones
6. Do not make up information not present in the provided context

FORMAT YOUR RESPONSES:
- Start with a direct answer
- List relevant tools with their sections
- Include parameter details when helpful
- Provide usage examples if applicable"""
        
        prompt = f"""{system_prompt}

CONTEXT (Available Tools):
{context}

USER QUESTION: {query}

Please provide a helpful answer based strictly on the available tools above."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            console.print(f"[red]✗ Failed to generate response: {e}[/red]")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

class GitHubMCPChatbot:
    """Main chatbot class for querying GitHub MCP tools"""
    
    def __init__(self, 
                 gemini_api_key: str,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.embedding_service = EmbeddingService(embedding_model)
        self.weaviate_service = WeaviateQueryService(weaviate_url, weaviate_api_key)
        self.gemini_service = GeminiService(gemini_api_key)
        
        # Check if data exists in Weaviate
        self._check_data_availability()
    
    def _check_data_availability(self):
        """Check if required data is available in Weaviate"""
        if not self.weaviate_service.check_data_exists():
            console.print(Panel.fit(
                "[red]⚠️  No data found in Weaviate![/red]\n\n"
                "Please run the data processor first:\n"
                "[yellow]python data_processor.py[/yellow]\n\n"
                "Or if you have a different JSON file:\n"
                "[yellow]python data_processor.py --json-file your_file.json[/yellow]",
                title="Data Not Found",
                border_style="red"
            ))
            raise ValueError("No data found in Weaviate. Please run data_processor.py first.")
    
    def query(self, user_query: str, top_k: int = 5) -> str:
        """Process a user query and return response"""
        try:
            # Generate embedding for user query
            query_embedding = self.embedding_service.embed_text(user_query)
            
            # Search for similar chunks
            similar_chunks = self.weaviate_service.search_similar(query_embedding, limit=top_k)
            
            if not similar_chunks:
                return "I couldn't find any relevant tools for your query. Please try rephrasing your question."
            
            # Generate response using Gemini
            response = self.gemini_service.generate_response(user_query, similar_chunks)
            
            return response
            
        except Exception as e:
            console.print(f"[red]✗ Query failed: {e}[/red]")
            return f"Sorry, I encountered an error processing your query: {str(e)}"
    
    def interactive_chat(self):
        """Start interactive chat loop"""
        console.print(Panel.fit("🚀 GitHub MCP Tools Chatbot Ready!", style="bold green"))
        console.print("\n[bold cyan]Ask me about GitHub MCP tools! Type 'quit' to exit.[/bold cyan]\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if not user_input.strip():
                    continue
                
                console.print("\n[yellow]🤔 Thinking...[/yellow]")
                response = self.query(user_input)
                
                console.print(f"\n[bold green]🤖 Bot:[/bold green]")
                console.print(Panel(Markdown(response), title="Response", border_style="green"))
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]✗ Error: {e}[/red]")
    
    def single_query(self, query: str) -> str:
        """Process a single query (useful for API/programmatic usage)"""
        return self.query(query)

def main():
    """Main function to run the chatbot"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub MCP Tools Chatbot")
    parser.add_argument("--query", help="Single query to process (non-interactive mode)")
    parser.add_argument("--top-k", type=int, default=5, 
                       help="Number of similar tools to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    # Configuration from environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Validate configuration
    if not GEMINI_API_KEY:
        console.print("[red]✗ Please set GEMINI_API_KEY environment variable[/red]")
        return
    
    if ".weaviate.cloud" in WEAVIATE_URL and not WEAVIATE_API_KEY:
        console.print("[red]✗ Please set WEAVIATE_API_KEY for Weaviate Cloud connection[/red]")
        return
    
    try:
        # Initialize chatbot
        chatbot = GitHubMCPChatbot(
            gemini_api_key=GEMINI_API_KEY,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY,
            embedding_model=EMBEDDING_MODEL
        )
        
        # Single query mode or interactive mode
        if args.query:
            console.print(f"[bold blue]Query:[/bold blue] {args.query}\n")
            response = chatbot.single_query(args.query)
            console.print(f"[bold green]Response:[/bold green]")
            console.print(Panel(Markdown(response), border_style="green"))
        else:
            # Start interactive chat
            chatbot.interactive_chat()
        
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {e}[/red]")
        logger.exception("Fatal error occurred")

if __name__ == "__main__":
    main()