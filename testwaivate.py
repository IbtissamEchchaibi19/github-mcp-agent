#!/usr/bin/env python3
"""
GitHub MCP Tools Chatbot
A RAG-based chatbot for querying GitHub MCP tools using Weaviate and Gemini.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wc
from weaviate.classes.query import Filter, MetadataQuery
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

@dataclass
class ToolChunk:
    """Represents a tool chunk with metadata"""
    section: str
    tool_name: str
    description: str
    parameters: Dict[str, Any]
    full_text: str
    chunk_id: str

class EmbeddingService:
    """Handles text embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            console.print(f"[yellow]Loading embedding model: {self.model_name}[/yellow]")
            self.model = SentenceTransformer(self.model_name)
            console.print(f"[green]✓ Embedding model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load embedding model: {e}[/red]")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(texts).tolist()

class WeaviateService:
    """Handles Weaviate vector database operations"""
    
    def __init__(self, url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.collection_name = "GitHubMCPTools"
        self._connect()
    
    def _connect(self):
        """Connect to Weaviate instance"""
        try:
            console.print(f"[yellow]Connecting to Weaviate at {self.url}[/yellow]")
            
            # Check if it's a cloud URL (contains .weaviate.cloud)
            if ".weaviate.cloud" in self.url:
                if not self.api_key:
                    raise ValueError("API key is required for Weaviate Cloud connections")
                
                # Connect to Weaviate Cloud
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                )
            else:
                # Connect to local Weaviate instance
                # Parse URL to extract host and port
                if self.url.startswith("http://"):
                    host_port = self.url.replace("http://", "")
                elif self.url.startswith("https://"):
                    host_port = self.url.replace("https://", "")
                else:
                    host_port = self.url
                
                if ":" in host_port:
                    host, port = host_port.split(":")
                    port = int(port)
                else:
                    host = host_port
                    port = 8080
                
                self.client = weaviate.connect_to_local(host=host, port=port)
            
            # Test connection
            if self.client.is_ready():
                console.print("[green]✓ Connected to Weaviate successfully[/green]")
            else:
                raise ConnectionError("Weaviate is not ready")
                
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to Weaviate: {e}[/red]")
            raise
    
    def __del__(self):
        """Ensure client is properly closed"""
        if self.client:
            self.client.close()
    
    def create_schema(self):
        """Create the schema for GitHub MCP tools"""
        try:
            # Delete existing collection if it exists
            if self.client.collections.exists(self.collection_name):
                console.print(f"[yellow]Deleting existing collection: {self.collection_name}[/yellow]")
                self.client.collections.delete(self.collection_name)
            
            # Create collection with schema for custom vectors
            console.print(f"[yellow]Creating collection: {self.collection_name}[/yellow]")
            
            collection = self.client.collections.create(
                name=self.collection_name,
                description="GitHub MCP Tools for RAG chatbot",
                properties=[
                    wc.Property(name="section", data_type=wc.DataType.TEXT, description="Tool section/category"),
                    wc.Property(name="toolName", data_type=wc.DataType.TEXT, description="Name of the tool"),
                    wc.Property(name="description", data_type=wc.DataType.TEXT, description="Tool description"),
                    wc.Property(name="parameters", data_type=wc.DataType.TEXT, description="Tool parameters as JSON string"),
                    wc.Property(name="fullText", data_type=wc.DataType.TEXT, description="Complete tool information for context"),
                    wc.Property(name="chunkId", data_type=wc.DataType.TEXT, description="Unique identifier for the chunk")
                ],
                # Configure for custom vectors (no built-in vectorizer)
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.COSINE
                )
            )
            
            console.print("[green]✓ Schema created successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to create schema: {e}[/red]")
            raise
    
    def store_chunks(self, chunks: List[ToolChunk], embeddings: List[List[float]]):
        """Store tool chunks with their embeddings"""
        try:
            console.print(f"[yellow]Storing {len(chunks)} chunks in Weaviate[/yellow]")
            
            collection = self.client.collections.get(self.collection_name)
            
            # Insert objects one by one with vectors
            inserted_count = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    properties = {
                        "section": chunk.section,
                        "toolName": chunk.tool_name,
                        "description": chunk.description,
                        "parameters": json.dumps(chunk.parameters),
                        "fullText": chunk.full_text,
                        "chunkId": chunk.chunk_id
                    }
                    
                    # Insert single object with vector
                    uuid = collection.data.insert(
                        properties=properties,
                        vector=embedding
                    )
                    
                    inserted_count += 1
                    
                    if (i + 1) % 20 == 0:
                        console.print(f"[blue]Processed {i + 1}/{len(chunks)} chunks[/blue]")
                        
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to insert chunk {i}: {e}[/yellow]")
                    continue
            
            console.print(f"[green]✓ Successfully stored {inserted_count}/{len(chunks)} chunks[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to store chunks: {e}[/red]")
            raise
    
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

class DataProcessor:
    """Processes GitHub MCP tools JSON data"""
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]✗ Failed to load JSON: {e}[/red]")
            raise
    
    @staticmethod
    def extract_chunks(data: Dict[str, Any]) -> List[ToolChunk]:
        """Extract tool chunks from JSON data"""
        chunks = []
        
        for section_name, section_content in data.items():
            # Skip non-tool sections
            if not isinstance(section_content, dict):
                continue
            
            # Handle empty sections or sections without tools
            if not section_content:
                continue
            
            for tool_name, tool_info in section_content.items():
                if not isinstance(tool_info, dict):
                    continue
                
                description = tool_info.get("description", "")
                parameters = tool_info.get("parameters", {})
                
                # Create full text for better context
                full_text = f"Section: {section_name}\n"
                full_text += f"Tool: {tool_name}\n"
                full_text += f"Description: {description}\n"
                
                if parameters:
                    full_text += "Parameters:\n"
                    for param_name, param_info in parameters.items():
                        if isinstance(param_info, str):
                            full_text += f"  - {param_name}: {param_info}\n"
                        else:
                            full_text += f"  - {param_name}: {json.dumps(param_info, indent=4)}\n"
                
                chunk_id = f"{section_name}_{tool_name}".replace(" ", "_").lower()
                
                chunk = ToolChunk(
                    section=section_name,
                    tool_name=tool_name,
                    description=description,
                    parameters=parameters,
                    full_text=full_text,
                    chunk_id=chunk_id
                )
                
                chunks.append(chunk)
        
        console.print(f"[green]✓ Extracted {len(chunks)} tool chunks[/green]")
        return chunks

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
    """Main chatbot class that orchestrates all services"""
    
    def __init__(self, 
                 json_file_path: str,
                 gemini_api_key: str,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.json_file_path = json_file_path
        self.embedding_service = EmbeddingService(embedding_model)
        self.weaviate_service = WeaviateService(weaviate_url, weaviate_api_key)
        self.gemini_service = GeminiService(gemini_api_key)
        self.data_processor = DataProcessor()
        
    def initialize(self):
        """Initialize the chatbot by processing data and storing in Weaviate"""
        console.print(Panel.fit("🤖 Initializing GitHub MCP Tools Chatbot", style="bold blue"))
        
        # Load and process data
        console.print("\n[bold]Step 1: Loading JSON data[/bold]")
        data = self.data_processor.load_json(self.json_file_path)
        
        console.print("\n[bold]Step 2: Extracting tool chunks[/bold]")
        chunks = self.data_processor.extract_chunks(data)
        
        console.print("\n[bold]Step 3: Generating embeddings[/bold]")
        texts = [chunk.full_text for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        
        console.print("\n[bold]Step 4: Setting up Weaviate schema[/bold]")
        self.weaviate_service.create_schema()
        
        console.print("\n[bold]Step 5: Storing chunks in Weaviate[/bold]")
        self.weaviate_service.store_chunks(chunks, embeddings)
        
        console.print(Panel.fit("✅ Initialization Complete!", style="bold green"))
    
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

def main():
    """Main function to run the chatbot"""
    
    # Configuration
    JSON_FILE_PATH = "github_mcp_tools.json"
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
    
    if not Path(JSON_FILE_PATH).exists():
        console.print(f"[red]✗ JSON file not found: {JSON_FILE_PATH}[/red]")
        return
    
    try:
        # Initialize chatbot
        chatbot = GitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            gemini_api_key=GEMINI_API_KEY,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY,
            embedding_model=EMBEDDING_MODEL
        )
        
        # Initialize (this processes data and stores in Weaviate)
        chatbot.initialize()
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {e}[/red]")
        logger.exception("Fatal error occurred")

if __name__ == "__main__":
    main()