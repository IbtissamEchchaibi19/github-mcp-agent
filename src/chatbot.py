import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from Weaviate import EnhancedGitHubMCPStorage
from prompts import SIMPLE_SYSTEM_PROMPT

# Import our modules
from models import ChatState
from mcp_client import MCPClient
from utils import (
    has_recent_tool_list, extract_task_from_query, 
    format_tool_details_with_collection_start, 
    show_help, check_environment
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGitHubMCPChatbot:
    """Enhanced chatbot with parameter collection and MCP execution"""
    
    def __init__(self, 
                 json_file_path: str,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None):
        
        # Initialize storage system
        self.storage = EnhancedGitHubMCPStorage(
            json_file_path=json_file_path,
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize memory
        self.memory = MemorySaver()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize MCP client
        self.mcp_client = MCPClient()
        
        # Store for tool selection memory
        self._last_tool_details = []
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        print("🤖 Enhanced GitHub MCP Tools Chatbot with Execution Initialized")
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the Groq LLM"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            model="meta-llama/llama-3.3-70b-versatile",
            temperature=0,
            max_completion_tokens=200,
            top_p=0.9,
            stream=False,
            api_key=api_key
        )
    
    def _create_tools(self) -> List:
        """Create tools"""
        
        @tool
        def search_github_tools(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Search for relevant GitHub MCP tools"""
            try:
                results = self.storage.search_tools(query, top_k=top_k)
                return results
            except Exception as e:
                logger.error(f"Tool search failed: {e}")
                return []
        
        @tool
        def get_tool_details(tool_name: str) -> Optional[Dict[str, Any]]:
            """Get detailed parameters for a specific tool"""
            try:
                return self.storage.get_tool_by_name(tool_name)
            except Exception as e:
                logger.error(f"Failed to get tool details: {e}")
                return None
        
        return [search_github_tools, get_tool_details]
    
    # ... (keep all your existing methods like _build_workflow, _process_input, etc.)
    # Just replace self._run_mcp_tool calls with self.mcp_client.run_mcp_tool
    # And use the utils functions where appropriate
    
    def run_interactive_session(self):
        """Run interactive session"""
        print("\n" + "="*60)
        print("🤖 ENHANCED GITHUB MCP TOOLS CHATBOT WITH EXECUTION")
        print("="*60)
        print("Search tools → Select tool → Provide parameters → Execute!")
        print("Type 'quit' to exit, 'help' for commands.")
        print("="*60)
        
        # Check environment variables
        check_environment()
        
        # Initialize storage
        try:
            stats = self.storage.get_storage_stats()
            if stats.get('total_tools', 0) == 0:
                print("Initializing storage...")
                self.storage.initialize_storage()
            print(f"Ready with {stats.get('total_tools', 0)} tools")
        except:
            print("Initializing storage...")
            self.storage.initialize_storage()
        
        thread_id = "interactive_session"
        
        while True:
            try:
                user_input = input("\n[You]: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    show_help()
                    continue
                
                if not user_input:
                    continue
                
                # Process message
                final_state = self.process_message(user_input, thread_id)
                
                # Get the last assistant message
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        print(f"\n[Bot]: {msg.content}")
                        break
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")