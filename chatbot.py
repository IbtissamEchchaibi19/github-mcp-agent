import os
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from Weaviate import EnhancedGitHubMCPStorage
from prompts import SIMPLE_SYSTEM_PROMPT
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State Definition
class ChatState(TypedDict):
    """State for the simplified chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    search_results: List[Dict[str, Any]]
    recommended_tools: List[str]
    tool_details: List[Dict[str, Any]]
    selected_tool: Optional[str]
    current_step: str
    error: Optional[str]

class SimplifiedGitHubMCPChatbot:
    """Simplified chatbot with direct responses"""
    
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
        
        # Store for tool selection memory
        self._last_tool_details = []
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        print("🤖 Simplified GitHub MCP Tools Chatbot Initialized")
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the Groq LLM with lower temperature for consistency"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,  # Lower temperature for more consistent responses
            max_completion_tokens=200,  # Limit response length
            top_p=0.9,
            stream=False,
            api_key=api_key
        )
    
    def _create_tools(self) -> List:
        """Create simplified tools"""
        
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
    
    def _build_workflow(self) -> StateGraph:
        """Build simplified workflow"""
        
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("input_processor", self._process_input)
        workflow.add_node("tool_searcher", self._search_tools)
        workflow.add_node("response_generator", self._generate_response)
        workflow.add_node("tool_details_provider", self._provide_tool_details)
        
        # Define workflow
        workflow.set_entry_point("input_processor")
        workflow.add_edge("input_processor", "tool_searcher")
        
        # Conditional routing based on input type
        workflow.add_conditional_edges(
            "tool_searcher",
            self._route_based_on_input,
            {
                "tool_selection": "tool_details_provider",
                "tool_search": "response_generator"
            }
        )
        
        workflow.add_edge("response_generator", END)
        workflow.add_edge("tool_details_provider", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _process_input(self, state: ChatState) -> ChatState:
        """Process user input"""
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if user_message:
            state["user_query"] = user_message
            state["current_step"] = "input_processed"
            
        return state
    
    def _route_based_on_input(self, state: ChatState) -> str:
        """Route based on whether user is selecting a tool or searching"""
        user_query = state["user_query"].strip()
        
        # Check if user is selecting a tool number
        if user_query.isdigit() or any(word in user_query.lower() 
                                     for word in ['choose', 'select', 'pick', 'number']):
            return "tool_selection"
        else:
            return "tool_search"
    
    def _search_tools(self, state: ChatState) -> ChatState:
        """Search for relevant tools"""
        try:
            user_query = state["user_query"]
            search_results = self.storage.search_tools(user_query, top_k=5)
            
            state["search_results"] = search_results
            state["current_step"] = "tools_searched"
            
        except Exception as e:
            state["error"] = str(e)
            logger.error(f"Tool search failed: {e}")
        
        return state
    
    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate simplified response with tool recommendations"""
        try:
            search_results = state.get("search_results", [])
            user_query = state["user_query"]
            
            if not search_results:
                response = f"No tools found for '{user_query}'. Try a different search term."
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Create simple tool list
            task = self._extract_task_from_query(user_query)
            response_lines = [f"Here are the relevant tools for {task}:", ""]
            
            for i, tool in enumerate(search_results[:5], 1):
                tool_name = tool.get('toolName', 'Unknown')
                description = tool.get('description', '')[:50] + "..." if len(tool.get('description', '')) > 50 else tool.get('description', '')
                response_lines.append(f"{i}. {tool_name} - {description}")
            
            response_lines.append("")
            response_lines.append(f"Choose a number (1-{min(len(search_results), 5)}) for detailed parameters.")
            
            response = "\n".join(response_lines)
            
            # Store tools for later selection - FIXED: Store in state AND instance variable
            state["tool_details"] = search_results[:5]
            self._last_tool_details = search_results[:5]  # Store for access across workflow runs
            
            assistant_message = AIMessage(content=response)
            state["messages"].append(assistant_message)
            state["current_step"] = "response_generated"
            
        except Exception as e:
            error_response = f"Error: {str(e)}"
            assistant_message = AIMessage(content=error_response)
            state["messages"].append(assistant_message)
        
        return state
    
    def _provide_tool_details(self, state: ChatState) -> ChatState:
        """Provide detailed parameters for selected tool"""
        try:
            user_query = state["user_query"].strip()
            
            # Extract tool selection
            tool_number = None
            if user_query.isdigit():
                tool_number = int(user_query)
            else:
                # Extract number from text
                import re
                numbers = re.findall(r'\d+', user_query)
                if numbers:
                    tool_number = int(numbers[0])
            
            # Get previous tool details from conversation
            tool_details = self._get_previous_tool_details(state["messages"])
            
            if not tool_details or not tool_number or tool_number < 1 or tool_number > len(tool_details):
                response = "Please provide a valid tool number from the previous list."
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Get selected tool
            selected_tool = tool_details[tool_number - 1]
            tool_name = selected_tool.get('toolName')
            
            # Get full tool details from Weaviate
            full_details = self.storage.get_tool_by_name(tool_name)
            
            if not full_details:
                response = f"Could not retrieve details for {tool_name}"
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Format tool details
            response = self._format_tool_details(full_details)
            
            assistant_message = AIMessage(content=response)
            state["messages"].append(assistant_message)
            state["current_step"] = "tool_details_provided"
            
        except Exception as e:
            error_response = f"Error retrieving tool details: {str(e)}"
            assistant_message = AIMessage(content=error_response)
            state["messages"].append(assistant_message)
        
        return state
    
    def _extract_task_from_query(self, query: str) -> str:
        """Extract task description from user query"""
        query_lower = query.lower()
        
        # Simple task extraction
        if 'repository' in query_lower or 'repo' in query_lower:
            return "repository management"
        elif 'issue' in query_lower:
            return "issue management"
        elif 'pull request' in query_lower or 'pr' in query_lower:
            return "pull request management"
        elif 'workflow' in query_lower or 'action' in query_lower:
            return "workflow management"
        elif 'user' in query_lower or 'member' in query_lower:
            return "user management"
        else:
            return query.lower()
    
    def _get_previous_tool_details(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Get tool details from conversation memory"""
        try:
            return getattr(self, '_last_tool_details', [])
        except:
            return []
    
    def _format_tool_details(self, tool_details: Dict[str, Any]) -> str:
        """Format tool details in a clean way"""
        lines = []
        lines.append(f"Tool: {tool_details.get('toolName', 'Unknown')}")
        lines.append(f"Section: {tool_details.get('section', 'Unknown')}")
        lines.append(f"Description: {tool_details.get('description', 'No description')}")
        lines.append("")
        
        # Required parameters
        required_params = tool_details.get('requiredParams', [])
        if required_params:
            lines.append("Required Parameters:")
            for param in required_params:
                lines.append(f"  - {param}")
            lines.append("")
        
        # Optional parameters
        optional_params = tool_details.get('optionalParams', [])
        if optional_params:
            lines.append("Optional Parameters:")
            for param in optional_params:
                lines.append(f"  - {param}")
            lines.append("")
        
        # Parameters schema if available
        params_schema = tool_details.get('parametersSchema')
        if params_schema:
            lines.append("Parameter Details:")
            try:
                import json
                if isinstance(params_schema, str):
                    schema = json.loads(params_schema)
                else:
                    schema = params_schema
                
                for param_name, param_info in schema.items():
                    if isinstance(param_info, str):
                        lines.append(f"  - {param_name}: {param_info}")
                    else:
                        lines.append(f"  - {param_name}: {str(param_info)[:100]}")
            except:
                lines.append(f"  {params_schema}")
        
        return "\n".join(lines)
    
    def process_message(self, user_input: str, thread_id: str = "default") -> ChatState:
        """Process user message"""
        initial_state = ChatState(
            messages=[HumanMessage(content=user_input)],
            user_query="",
            search_results=[],
            recommended_tools=[],
            tool_details=[],
            selected_tool=None,
            current_step="",
            error=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.workflow.invoke(initial_state, config)
        
        return final_state
    
    def run_interactive_session(self):
        """Run interactive session with simplified responses"""
        print("\n" + "="*50)
        print("🤖 SIMPLIFIED GITHUB MCP TOOLS CHATBOT")
        print("="*50)
        print("Ask for tools and get direct recommendations.")
        print("Type 'quit' to exit, 'help' for commands.")
        print("="*50)
        
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
                    self._show_simple_help()
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
    
    def _show_simple_help(self):
        """Show simplified help"""
        print("\nCommands:")
        print("• Ask about tools: 'tools for repository management'")
        print("• Select tool: Type number from list (e.g., '2')")
        print("• quit - Exit")

def main():
    """Main function"""
    JSON_FILE_PATH = "github_mcp_tools.json"
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    try:
        chatbot = SimplifiedGitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY
        )
        
        chatbot.run_interactive_session()
        
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main()