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
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel

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
    
    def collection_exists(self) -> bool:
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

class GitHubMCPDataProcessor:
    """Main class for processing and storing GitHub MCP tools data"""
    
    def __init__(self, 
                 json_file_path: str,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.json_file_path = json_file_path
        self.embedding_service = EmbeddingService(embedding_model)
        self.weaviate_service = WeaviateService(weaviate_url, weaviate_api_key)
        self.data_processor = DataProcessor()
    
    def process_and_store(self, force_recreate: bool = False):
        """Process JSON data and store in Weaviate"""
        console.print(Panel.fit("📊 Processing GitHub MCP Tools Data", style="bold blue"))
        
        # Check if data already exists
        if not force_recreate and self.weaviate_service.collection_exists():
            console.print("[green]✓ Data already exists in Weaviate. Use --force to recreate.[/green]")
            return
        
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
        
        console.print(Panel.fit("✅ Data Processing Complete!", style="bold green"))

def main():
    """Main function to run data processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and store GitHub MCP tools data")
    parser.add_argument("--json-file", default="github_mcp_tools.json", 
                       help="Path to JSON file containing GitHub MCP tools data")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreation of data even if it already exists")
    
    args = parser.parse_args()
    
    # Configuration from environment variables
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Validate configuration
    if ".weaviate.cloud" in WEAVIATE_URL and not WEAVIATE_API_KEY:
        console.print("[red]✗ Please set WEAVIATE_API_KEY for Weaviate Cloud connection[/red]")
        return
    
    if not Path(args.json_file).exists():
        console.print(f"[red]✗ JSON file not found: {args.json_file}[/red]")
        return
    
    try:
        # Initialize processor
        processor = GitHubMCPDataProcessor(
            json_file_path=args.json_file,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY,
            embedding_model=EMBEDDING_MODEL
        )
        
        # Process and store data
        processor.process_and_store(force_recreate=args.force)
        
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {e}[/red]")
        logger.exception("Fatal error occurred")
