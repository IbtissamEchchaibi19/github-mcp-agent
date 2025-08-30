import os
import json
import subprocess
import time
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Tuple
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from Weaviate import EnhancedGitHubMCPStorage
import gradio as gr

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State Definition
class ChatState(TypedDict):
    """Enhanced state for the chatbot with MCP execution"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    search_results: List[Dict[str, Any]]
    recommended_tools: List[str]
    tool_details: List[Dict[str, Any]]
    selected_tool: Optional[Dict[str, Any]]
    collected_parameters: Dict[str, Any]
    current_step: str
    error: Optional[str]
    mcp_process: Optional[subprocess.Popen]
    mcp_response: Optional[Dict[str, Any]]
    awaiting_parameters: bool
    parameter_collection_state: Dict[str, Any]
    tool_selection_mode: bool  # NEW: Track if we're in tool selection mode

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
        
        # Store for tool selection memory
        self._last_tool_details = []
        
        # MCP server request ID counter
        self.request_id = 1
        
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
    
    def _build_workflow(self) -> StateGraph:
        """Build enhanced workflow with parameter collection and execution"""
        
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("input_processor", self._process_input)
        workflow.add_node("tool_searcher", self._search_tools)
        workflow.add_node("response_generator", self._generate_response)
        workflow.add_node("tool_details_provider", self._provide_tool_details)
        workflow.add_node("parameter_collector", self._collect_parameters)
        workflow.add_node("mcp_executor", self._execute_mcp_tool)
        workflow.add_node("result_formatter", self._format_execution_result)
        
        # Define workflow
        workflow.set_entry_point("input_processor")
        workflow.add_edge("input_processor", "tool_searcher")
        
        # Conditional routing based on input type
        workflow.add_conditional_edges(
            "tool_searcher",
            self._route_based_on_input,
            {
                "tool_selection": "tool_details_provider",
                "tool_search": "response_generator",
                "parameter_input": "parameter_collector",
                "execute_tool": "mcp_executor"
            }
        )
        
        workflow.add_edge("response_generator", END)
        workflow.add_edge("tool_details_provider", END)  # CHANGED: Don't auto-collect parameters
        workflow.add_edge("parameter_collector", END)
        workflow.add_edge("mcp_executor", "result_formatter")
        workflow.add_edge("result_formatter", END)
        
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
        """Enhanced routing based on conversation state"""
        user_query = state["user_query"].strip().lower()
        
        print(f"DEBUG: Routing - User query: '{user_query}'")  # Debug
        print(f"DEBUG: Routing - Awaiting parameters: {state.get('awaiting_parameters', False)}")  # Debug
        print(f"DEBUG: Routing - Tool selection mode: {state.get('tool_selection_mode', False)}")  # Debug
        print(f"DEBUG: Routing - Selected tool: {state.get('selected_tool') is not None}")  # Debug
        
        # Check if we're in parameter collection mode
        if state.get("awaiting_parameters", False):
            return "parameter_input"
        
        # Check if user wants to execute after parameter collection
        if state.get("selected_tool") and any(word in user_query for word in ['execute', 'run', 'yes', 'go']):
            return "execute_tool"
        
        # Check if we're in tool selection mode (just showed tools list)
        if state.get("tool_selection_mode", False) and user_query.isdigit():
            tool_number = int(user_query)
            tool_details = state.get("tool_details", [])
            if 1 <= tool_number <= len(tool_details):
                return "tool_selection"
        
        # Check if user is selecting a tool number from previous results
        # Look for recent tool list in conversation
        if self._has_recent_tool_list(state["messages"]) and user_query.isdigit():
            tool_number = int(user_query)
            if 1 <= tool_number <= 5:  # Valid tool selection range
                return "tool_selection"
        
        # Default to tool search
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
        """Generate response with tool recommendations"""
        try:
            search_results = state.get("search_results", [])
            user_query = state["user_query"]
            
            if not search_results:
                response = f"No tools found for '{user_query}'. Try a different search term."
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Create tool list
            task = self._extract_task_from_query(user_query)
            response_lines = [f"Here are the relevant tools for {task}:", ""]
            
            for i, tool in enumerate(search_results[:5], 1):
                tool_name = tool.get('toolName', 'Unknown')
                description = tool.get('description', '')[:50] + "..." if len(tool.get('description', '')) > 50 else tool.get('description', '')
                response_lines.append(f"{i}. {tool_name} - {description}")
            
            response_lines.append("")
            response_lines.append(f"Choose a number (1-{min(len(search_results), 5)}) to see parameters and execute the tool.")
            
            response = "\n".join(response_lines)
            
            # Store tools for later selection and set selection mode
            state["tool_details"] = search_results[:5]
            state["tool_selection_mode"] = True  # NEW: Set tool selection mode
            self._last_tool_details = search_results[:5]
            
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
            
            print(f"DEBUG: Tool details - User query: '{user_query}'")  # Debug
            
            # Extract tool selection number
            tool_number = None
            if user_query.isdigit():
                tool_number = int(user_query)
            else:
                import re
                numbers = re.findall(r'\d+', user_query)
                if numbers:
                    tool_number = int(numbers[0])
            
            print(f"DEBUG: Tool details - Tool number: {tool_number}")  # Debug
            
            # Get tool details from the last search results stored in state
            tool_details = state.get("tool_details", [])
            if not tool_details:
                # Fallback to instance variable
                tool_details = getattr(self, '_last_tool_details', [])
            
            print(f"DEBUG: Tool details - Available tools: {len(tool_details)}")  # Debug
            
            if not tool_details or not tool_number or tool_number < 1 or tool_number > len(tool_details):
                response = f"Please provide a valid tool number (1-{len(tool_details) if tool_details else 0}) from the previous list."
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Get selected tool
            selected_tool = tool_details[tool_number - 1]
            tool_name = selected_tool.get('toolName')
            
            print(f"DEBUG: Selected tool: {tool_name}")  # Debug
            
            # Get full tool details
            full_details = self.storage.get_tool_by_name(tool_name)
            
            if not full_details:
                response = f"Could not retrieve details for {tool_name}"
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            # Store selected tool in state
            state["selected_tool"] = full_details
            state["collected_parameters"] = {}
            state["awaiting_parameters"] = True
            state["tool_selection_mode"] = False  # NEW: Clear tool selection mode
            
            # Initialize parameter collection state
            required_params = full_details.get('requiredParams', [])
            optional_params = full_details.get('optionalParams', [])
            
            print(f"DEBUG: Required params: {required_params}")  # Debug output
            print(f"DEBUG: Optional params: {optional_params}")  # Debug output
            
            state["parameter_collection_state"] = {
                "required_params": required_params,
                "optional_params": optional_params,
                "current_param_index": 0,
                "collecting_required": True
            }
            
            # Format initial response with tool details
            response = self._format_tool_details_with_collection_start(full_details)
            
            assistant_message = AIMessage(content=response)
            state["messages"].append(assistant_message)
            state["current_step"] = "tool_details_provided"
            
        except Exception as e:
            print(f"DEBUG: Error in _provide_tool_details: {e}")  # Debug output
            error_response = f"Error retrieving tool details: {str(e)}"
            assistant_message = AIMessage(content=error_response)
            state["messages"].append(assistant_message)
        
        return state
    
    def _collect_parameters(self, state: ChatState) -> ChatState:
        """Collect parameters from user input"""
        try:
            user_input = state["user_query"].strip()
            collection_state = state.get("parameter_collection_state", {})
            collected_params = state.get("collected_parameters", {})
            
            print(f"DEBUG: Parameter collection - User input: '{user_input}'")  # Debug
            print(f"DEBUG: Parameter collection - Collection state: {collection_state}")  # Debug
            print(f"DEBUG: Parameter collection - Collected params: {collected_params}")  # Debug
            
            # Get current parameter info
            required_params = collection_state.get("required_params", [])
            optional_params = collection_state.get("optional_params", [])
            current_index = collection_state.get("current_param_index", 0)
            collecting_required = collection_state.get("collecting_required", True)
            
            if collecting_required and current_index < len(required_params):
                # Collecting required parameters
                current_param = required_params[current_index]
                
                # Store the parameter value
                collected_params[current_param] = user_input
                
                # Move to next parameter
                current_index += 1
                
                if current_index < len(required_params):
                    # Ask for next required parameter
                    next_param = required_params[current_index]
                    response = f"✅ Got {current_param}: {user_input}\n\nNext required parameter: {next_param}\nPlease enter the value:"
                else:
                    # All required params collected, ask about optional
                    if optional_params:
                        current_index = 0
                        collecting_required = False
                        response = f"✅ Got {current_param}: {user_input}\n\nAll required parameters collected! 🎉\n\nOptional parameters available: {', '.join(optional_params)}\n\nDo you want to provide optional parameters? (yes/no) or type 'execute' to run the tool now:"
                    else:
                        # No optional params, ready to execute
                        response = f"✅ Got {current_param}: {user_input}\n\nAll parameters collected! 🎉\n\nReady to execute {state['selected_tool'].get('toolName')}. Type 'execute' to run the tool:"
                        state["awaiting_parameters"] = False
            
            elif not collecting_required:
                # Handle optional parameter collection
                if user_input.lower() in ['no', 'skip', 'execute']:
                    response = f"Ready to execute {state['selected_tool'].get('toolName')}! Type 'execute' to run:"
                    state["awaiting_parameters"] = False
                elif user_input.lower() == 'yes':
                    if current_index < len(optional_params):
                        next_param = optional_params[current_index]
                        response = f"Optional parameter: {next_param}\nPlease enter the value (or 'skip' to skip):"
                    else:
                        response = f"All optional parameters processed. Type 'execute' to run the tool:"
                        state["awaiting_parameters"] = False
                else:
                    # Collecting optional parameter value
                    if current_index < len(optional_params):
                        current_param = optional_params[current_index]
                        if user_input.lower() != 'skip':
                            collected_params[current_param] = user_input
                        
                        current_index += 1
                        
                        if current_index < len(optional_params):
                            next_param = optional_params[current_index]
                            response = f"✅ Got {current_param}: {user_input if user_input.lower() != 'skip' else 'skipped'}\n\nNext optional parameter: {next_param}\nPlease enter the value (or 'skip' to skip):"
                        else:
                            response = f"All parameters collected! Type 'execute' to run the tool:"
                            state["awaiting_parameters"] = False
            else:
                # Shouldn't reach here, but handle gracefully
                response = "Parameter collection completed. Type 'execute' to run the tool:"
                state["awaiting_parameters"] = False
            
            # Update state
            state["collected_parameters"] = collected_params
            state["parameter_collection_state"] = {
                "required_params": required_params,
                "optional_params": optional_params,
                "current_param_index": current_index,
                "collecting_required": collecting_required
            }
            
            assistant_message = AIMessage(content=response)
            state["messages"].append(assistant_message)
            
        except Exception as e:
            print(f"DEBUG: Error in parameter collection: {e}")  # Debug
            error_response = f"Error collecting parameters: {str(e)}"
            assistant_message = AIMessage(content=error_response)
            state["messages"].append(assistant_message)
        
        return state
    
    def _execute_mcp_tool(self, state: ChatState) -> ChatState:
        """Execute the GitHub MCP tool"""
        try:
            selected_tool = state.get("selected_tool")
            collected_parameters = state.get("collected_parameters", {})
            
            if not selected_tool:
                response = "No tool selected for execution."
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
                return state
            
            tool_name = selected_tool.get("toolName")
            
            # Connect to MCP server and execute
            response = f"🚀 Executing {tool_name} with parameters:\n{json.dumps(collected_parameters, indent=2)}\n\n"
            
            # Execute MCP tool
            mcp_result = self._run_mcp_tool(tool_name, collected_parameters)
            
            if mcp_result.get("success"):
                state["mcp_response"] = mcp_result["response"]
                response += "✅ Tool executed successfully!"
            else:
                response += f"❌ Tool execution failed: {mcp_result.get('error', 'Unknown error')}"
            
            assistant_message = AIMessage(content=response)
            state["messages"].append(assistant_message)
            
            # Reset state for next interaction
            state["selected_tool"] = None
            state["collected_parameters"] = {}
            state["awaiting_parameters"] = False
            state["parameter_collection_state"] = {}
            state["tool_selection_mode"] = False
            
        except Exception as e:
            error_response = f"Error executing tool: {str(e)}"
            assistant_message = AIMessage(content=error_response)
            state["messages"].append(assistant_message)
        
        return state
    
    def _format_execution_result(self, state: ChatState) -> ChatState:
        """Format the execution result"""
        try:
            mcp_response = state.get("mcp_response")
            
            if mcp_response:
                if "content" in mcp_response:
                    content_text = ""
                    for content_item in mcp_response["content"]:
                        if content_item.get("type") == "text":
                            content_text += content_item.get("text", "")
                    
                    response = f"📋 Result:\n{content_text}"
                else:
                    response = f"📋 Result:\n{json.dumps(mcp_response, indent=2)}"
                
                assistant_message = AIMessage(content=response)
                state["messages"].append(assistant_message)
            
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
        
        return state
    
    def _run_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run MCP tool and return result"""
        try:
            # Start GitHub MCP server
            process = subprocess.Popen(
                ['docker', 'run', '-i', '--rm', '-e', 'GITHUB_PERSONAL_ACCESS_TOKEN', 'ghcr.io/github/github-mcp-server'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env={**os.environ, 'GITHUB_PERSONAL_ACCESS_TOKEN': os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN', '')}
            )
            
            time.sleep(3)
            
            if process.poll() is not None:
                stderr_output = process.stderr.read()
                return {"success": False, "error": f"MCP server failed to start: {stderr_output}"}
            
            # Initialize server
            if not self._initialize_mcp_server(process):
                process.terminate()
                return {"success": False, "error": "Failed to initialize MCP server"}
            
            # Execute tool
            params = {
                "name": tool_name,
                "arguments": parameters
            }
            
            response = self._send_mcp_request(process, "tools/call", params)
            
            process.terminate()
            
            if response and "result" in response:
                return {"success": True, "response": response["result"]}
            elif response and "error" in response:
                error = response["error"]
                return {"success": False, "error": error.get('message', 'Unknown MCP error')}
            else:
                return {"success": False, "error": f"Unexpected MCP response: {response}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _initialize_mcp_server(self, process: subprocess.Popen) -> bool:
        """Initialize MCP server"""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "github-mcp-chatbot",
                "version": "1.0.0"
            }
        }
        
        response = self._send_mcp_request(process, "initialize", params)
        if response and "result" in response:
            self._send_mcp_request(process, "initialized", {})
            return True
        return False
    
    def _send_mcp_request(self, process: subprocess.Popen, method: str, params=None) -> Optional[Dict]:
        """Send JSON-RPC request to MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        
        if params:
            request["params"] = params
        
        self.request_id += 1
        
        try:
            request_json = json.dumps(request) + '\n'
            process.stdin.write(request_json)
            process.stdin.flush()
            
            response_line = process.stdout.readline()
            if not response_line:
                return None
            
            return json.loads(response_line.strip())
            
        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            return None
    
    def _has_recent_tool_list(self, messages: List[BaseMessage]) -> bool:
        """Check if the last bot message contains a tool list"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content.lower()
                # Check for tool list indicators
                if ("choose a number" in content and 
                    any(str(i) + "." in content for i in range(1, 6))):
                    return True
                break
        return False
    
    def _extract_task_from_query(self, query: str) -> str:
        """Extract task description from user query"""
        query_lower = query.lower()
        
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
    
    def _format_tool_details_with_collection_start(self, tool_details: Dict[str, Any]) -> str:
        """Format tool details and start parameter collection"""
        lines = []
        lines.append(f"🔧 Tool: {tool_details.get('toolName', 'Unknown')}")
        lines.append(f"📂 Section: {tool_details.get('section', 'Unknown')}")
        lines.append(f"📝 Description: {tool_details.get('description', 'No description')}")
        lines.append("")
        
        # Required parameters
        required_params = tool_details.get('requiredParams', [])
        if required_params:
            lines.append("✅ Required Parameters:")
            for param in required_params:
                lines.append(f"  • {param}")
            lines.append("")
        
        # Optional parameters
        optional_params = tool_details.get('optionalParams', [])
        if optional_params:
            lines.append("🔹 Optional Parameters:")
            for param in optional_params:
                lines.append(f"  • {param}")
            lines.append("")
        
        lines.append("🚀 Let's collect the parameters!")
        lines.append("")
        
        if required_params:
            lines.append(f"First required parameter: **{required_params[0]}**")
            lines.append("Please enter the value:")
        else:
            lines.append("No required parameters. Type 'execute' to run the tool or provide optional parameters.")
        
        return "\n".join(lines)
    
    def process_message(self, user_input: str, thread_id: str = "default") -> ChatState:
        """Process user message"""
        # Get current state from memory to preserve context
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Try to get existing state from memory
            existing_state = self.workflow.get_state(config)
            if existing_state and existing_state.values:
                current_state = existing_state.values
                # Add new message
                current_state["messages"].append(HumanMessage(content=user_input))
                current_state["user_query"] = user_input
            else:
                # Create new state
                current_state = ChatState(
                    messages=[HumanMessage(content=user_input)],
                    user_query=user_input,
                    search_results=[],
                    recommended_tools=[],
                    tool_details=[],
                    selected_tool=None,
                    collected_parameters={},
                    current_step="",
                    error=None,
                    mcp_process=None,
                    mcp_response=None,
                    awaiting_parameters=False,
                    parameter_collection_state={},
                    tool_selection_mode=False  # NEW: Initialize tool selection mode
                )
        except:
            # Fallback to new state if memory access fails
            current_state = ChatState(
                messages=[HumanMessage(content=user_input)],
                user_query=user_input,
                search_results=[],
                recommended_tools=[],
                tool_details=[],
                selected_tool=None,
                collected_parameters={},
                current_step="",
                error=None,
                mcp_process=None,
                mcp_response=None,
                awaiting_parameters=False,
                parameter_collection_state={},
                tool_selection_mode=False  # NEW: Initialize tool selection mode
            )
        
        final_state = self.workflow.invoke(current_state, config)
        return final_state
    
    def run_interactive_session(self):
        """Run interactive session"""
        print("\n" + "="*60)
        print("🤖 ENHANCED GITHUB MCP TOOLS CHATBOT WITH EXECUTION")
        print("="*60)
        print("Search tools → Select tool → Provide parameters → Execute!")
        print("Type 'quit' to exit, 'help' for commands.")
        print("="*60)
        
        # Check environment variables
        if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            print("⚠️  Warning: GITHUB_PERSONAL_ACCESS_TOKEN not set!")
            print("Set it with: export GITHUB_PERSONAL_ACCESS_TOKEN=your_token")
            print()
        
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
                    self._show_enhanced_help()
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
    
    def _show_enhanced_help(self):
        """Show enhanced help"""
        print("\n" + "="*50)
        print("📖 HELP - Enhanced GitHub MCP Chatbot")
        print("="*50)
        print("1️⃣  Search for tools:")
        print("   • 'tools for creating issues'")
        print("   • 'repository management tools'")
        print("   • 'pull request tools'")
        print("")
        print("2️⃣  Select a tool by number:")
        print("   • Type '1', '2', etc. from the list")
        print("")
        print("3️⃣  Provide parameters:")
        print("   • Enter values when prompted")
        print("   • Type 'skip' for optional parameters")
        print("")
        print("4️⃣  Execute the tool:")
        print("   • Type 'execute' to run")
        print("")
        print("⚠️  Requirements:")
        print("   • GROQ_API_KEY environment variable")
        print("   • GITHUB_PERSONAL_ACCESS_TOKEN environment variable")
        print("   • Docker installed and running")
        print("")
        print("Commands: 'quit' to exit, 'help' for this help")
        print("="*50)

    # NEW: Gradio Interface Methods
    def create_gradio_interface(self):
        """Create simple Gradio interface for the chatbot"""
        # Initialize storage if needed
        try:
            stats = self.storage.get_storage_stats()
            if stats.get('total_tools', 0) == 0:
                print("Initializing storage...")
                self.storage.initialize_storage()
        except:
            print("Initializing storage...")
            self.storage.initialize_storage()
        
        def chat_with_bot(message: str, history: List[Tuple[str, str]], thread_id: str) -> Tuple[str, List[Tuple[str, str]]]:
            """Process message through the chatbot and return response"""
            try:
                if not message.strip():
                    return "", history
                
                # Process message
                final_state = self.process_message(message, thread_id)
                
                # Get the last assistant message
                bot_response = "I'm processing your request..."
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        bot_response = msg.content
                        break
                
                # Add to history
                history.append((message, bot_response))
                
                return "", history
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                history.append((message, error_message))
                return "", history
        
        # Create simple interface
        with gr.Blocks(title="GitHub MCP Chatbot") as interface:
            
            gr.Markdown("# GitHub MCP Tools Chatbot")
            
            # Simple chat interface
            chatbot = gr.Chatbot(height=400)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask for tools (e.g., 'tools for issues')",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1)
            
            clear_btn = gr.Button("Clear")
            
            # Thread ID state (hidden)
            thread_id_state = gr.State("gradio_session")
            
            # Event handlers
            msg_input.submit(
                chat_with_bot,
                inputs=[msg_input, chatbot, thread_id_state],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                chat_with_bot,
                inputs=[msg_input, chatbot, thread_id_state],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg_input]
            )
        
        return interface
    
    def _get_status_message(self) -> str:
        """Get simple status message"""
        status = "Ready"
        if not os.getenv("GROQ_API_KEY"):
            status = "Missing GROQ_API_KEY"
        elif not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            status = "Missing GITHUB_TOKEN"
        return status
    
    def launch_gradio(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_gradio_interface()
        
        # Default launch parameters
        launch_params = {
            "share": False,
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True,
            "quiet": False
        }
        
        # Override with any provided parameters
        launch_params.update(kwargs)
        
        print("\n" + "="*60)
        print("🚀 LAUNCHING GRADIO INTERFACE")
        print("="*60)
        print("🌐 Access the web interface at:")
        print(f"   Local: http://localhost:{launch_params['server_port']}")
        if launch_params.get('share'):
            print("   Public: Link will be shown after launch")
        print("="*60)
        
        return interface.launch(**launch_params)

def main():
    """Main function with option to launch Gradio or terminal interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced GitHub MCP Tools Chatbot")
    parser.add_argument(
        "--interface", 
        choices=["gradio", "terminal"], 
        default="gradio",
        help="Choose interface type (default: gradio)"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create shareable Gradio link"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port for Gradio interface (default: 7860)"
    )
    
    args = parser.parse_args()
    
    JSON_FILE_PATH = "github_mcp_tools.json"
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    try:
        chatbot = EnhancedGitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY
        )
        
        if args.interface == "gradio":
            # Launch Gradio interface
            chatbot.launch_gradio(
                share=args.share,
                server_port=args.port
            )
        else:
            # Launch terminal interface
            chatbot.run_interactive_session()
            
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main()