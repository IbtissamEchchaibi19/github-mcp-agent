import os
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import your existing storage system
from Weaviate import EnhancedGitHubMCPStorage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# ==================== State Definition ====================

class ChatState(TypedDict):
    """State for the LangGraph chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: str
    search_results: List[Dict[str, Any]]
    recommended_tools: List[str]
    tool_details: List[Dict[str, Any]]
    reasoning: str
    follow_up_questions: List[str]
    conversation_context: Dict[str, Any]
    current_step: str
    error: Optional[str]

# ==================== Pydantic Models ====================

@dataclass
class ToolRecommendation:
    """Structure for tool recommendations"""
    tool_name: str
    section: str
    description: str
    action_type: str
    resource_type: str
    confidence_score: float
    reasoning: str
    required_params: List[str]
    optional_params: List[str]
    use_cases: List[str]
    complexity_level: str

class ChatbotResponse(BaseModel):
    """Structured response from the chatbot"""
    intent: str = Field(description="The detected user intent")
    recommended_tools: List[str] = Field(description="List of recommended tool names")
    reasoning: str = Field(description="Explanation of why these tools were recommended")
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

# ==================== LangGraph GitHub MCP Chatbot ====================

class LangGraphGitHubMCPChatbot:
    """LangGraph-based intelligent chatbot for GitHub MCP Tools"""
    
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
        
        # Initialize memory for conversation persistence
        self.memory = MemorySaver()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # System prompt for the LLM
        self.system_prompt = self._create_system_prompt()
        
        console.print(Panel.fit("🤖 LangGraph GitHub MCP Tools Chatbot Initialized", style="bold blue"))
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the Groq LLM"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
            max_completion_tokens=1024,
            top_p=0.9,
            stream=False,
            api_key=api_key
        )
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for tool recommendation"""
        return """You are an intelligent assistant specialized in GitHub MCP (Model Context Protocol) tools. Your role is to help users find the most appropriate GitHub tools based on their requests.

CAPABILITIES:
- Analyze user requests to understand their GitHub-related intent
- Recommend relevant GitHub MCP tools from the available collection
- Provide clear explanations for tool recommendations
- Suggest follow-up questions to better understand user needs

TOOL CATEGORIES AVAILABLE:
- Issues: Creating, listing, updating, commenting on GitHub issues
- Pull Requests: Managing PRs, reviews, merging, comments
- Repositories: Creating, managing, searching repositories
- Workflows: GitHub Actions, CI/CD pipelines, workflow management
- Users/Organizations: User management, team operations
- Security: Vulnerability management, security advisories
- Notifications: Managing GitHub notifications
- Gists: Code snippet management
- Discussions: Repository discussions and forums

RESPONSE GUIDELINES:
1. Always provide 1-3 most relevant tool recommendations
2. Explain WHY each tool is recommended for the user's specific need
3. Include complexity level and required parameters when relevant
4. Suggest follow-up questions to clarify ambiguous requests
5. Be concise but informative in your explanations

Remember: Focus on practical tool recommendations that directly address the user's GitHub workflow needs."""

    def _create_tools(self) -> List:
        """Create LangChain tools for the chatbot"""
        
        @tool
        def search_github_tools(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Search for relevant GitHub MCP tools based on a query"""
            try:
                results = self.storage.search_tools(query, top_k=top_k)
                return results
            except Exception as e:
                logger.error(f"Tool search failed: {e}")
                return []
        
        @tool
        def get_tool_details(tool_name: str) -> Optional[Dict[str, Any]]:
            """Get detailed information about a specific GitHub MCP tool"""
            try:
                return self.storage.get_tool_by_name(tool_name)
            except Exception as e:
                logger.error(f"Failed to get tool details: {e}")
                return None
        
        @tool
        def get_storage_stats() -> Dict[str, Any]:
            """Get statistics about the GitHub MCP tools storage"""
            try:
                return self.storage.get_storage_stats()
            except Exception as e:
                logger.error(f"Failed to get storage stats: {e}")
                return {"error": str(e)}
        
        @tool
        def get_tools_by_category(category: str) -> List[Dict[str, Any]]:
            """Get all tools in a specific category (section)"""
            try:
                # This would need to be implemented in your storage system
                # For now, we'll search by category
                return self.storage.search_tools(category, top_k=20)
            except Exception as e:
                logger.error(f"Failed to get tools by category: {e}")
                return []
        
        return [search_github_tools, get_tool_details, get_storage_stats, get_tools_by_category]
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the workflow graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("input_processor", self._process_input)
        workflow.add_node("intent_analyzer", self._analyze_intent)
        workflow.add_node("tool_searcher", self._search_tools)
        workflow.add_node("recommendation_generator", self._generate_recommendations)
        workflow.add_node("response_formatter", self._format_response)
        workflow.add_node("tool_executor", ToolNode(self.tools))
        workflow.add_node("error_handler", self._handle_error)
        
        # Define the workflow edges
        workflow.set_entry_point("input_processor")
        
        # Main workflow path
        workflow.add_edge("input_processor", "intent_analyzer")
        workflow.add_edge("intent_analyzer", "tool_searcher")
        workflow.add_edge("tool_searcher", "recommendation_generator")
        workflow.add_edge("recommendation_generator", "response_formatter")
        workflow.add_edge("response_formatter", END)
        
        # Add conditional edges for tool execution and error handling
        workflow.add_conditional_edges(
            "intent_analyzer",
            self._should_use_tools,
            {
                "use_tools": "tool_executor",
                "continue": "tool_searcher"
            }
        )
        
        workflow.add_edge("tool_executor", "recommendation_generator")
        
        # Error handling edges
        workflow.add_conditional_edges(
            "tool_searcher",
            self._check_for_errors,
            {
                "error": "error_handler",
                "continue": "recommendation_generator"
            }
        )
        
        workflow.add_edge("error_handler", END)
        
        # Compile the workflow with memory
        return workflow.compile(checkpointer=self.memory)
    
    # ==================== Node Functions ====================
    
    def _process_input(self, state: ChatState) -> ChatState:
        """Process user input and update state"""
        logger.info("Processing user input...")
        
        # Get the latest human message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if user_message:
            state["user_query"] = user_message
            state["current_step"] = "input_processed"
            
        return state
    
    def _analyze_intent(self, state: ChatState) -> ChatState:
        """Analyze user intent"""
        logger.info("Analyzing user intent...")
        
        user_query = state["user_query"]
        
        # Simple intent analysis (you can make this more sophisticated)
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ['create', 'make', 'new', 'add']):
            intent = 'create'
        elif any(word in query_lower for word in ['list', 'show', 'view', 'get', 'find all']):
            intent = 'list'
        elif any(word in query_lower for word in ['update', 'edit', 'modify', 'change']):
            intent = 'update'
        elif any(word in query_lower for word in ['search', 'find', 'look for']):
            intent = 'search'
        elif any(word in query_lower for word in ['delete', 'remove', 'close']):
            intent = 'delete'
        elif any(word in query_lower for word in ['details', 'info', 'information', 'tell me about']):
            intent = 'details'
        elif any(word in query_lower for word in ['help', 'commands', 'what can']):
            intent = 'help'
        elif any(word in query_lower for word in ['stats', 'statistics', 'overview']):
            intent = 'stats'
        else:
            intent = 'general'
        
        state["intent"] = intent
        state["current_step"] = "intent_analyzed"
        
        return state
    
    def _should_use_tools(self, state: ChatState) -> str:
        """Decide whether to use tools based on intent"""
        intent = state["intent"]
        
        # Use tools for specific intents
        if intent in ['stats', 'details']:
            return "use_tools"
        else:
            return "continue"
    
    def _search_tools(self, state: ChatState) -> ChatState:
        """Search for relevant tools"""
        logger.info("Searching for relevant tools...")
        
        try:
            user_query = state["user_query"]
            search_results = self.storage.search_tools(user_query, top_k=8)
            
            state["search_results"] = search_results
            state["current_step"] = "tools_searched"
            
        except Exception as e:
            state["error"] = str(e)
            logger.error(f"Tool search failed: {e}")
        
        return state
    
    def _generate_recommendations(self, state: ChatState) -> ChatState:
        """Generate tool recommendations using LLM"""
        logger.info("Generating recommendations...")
        
        try:
            search_results = state.get("search_results", [])
            user_query = state["user_query"]
            intent = state["intent"]
            
            if not search_results:
                state["reasoning"] = "I couldn't find specific tools matching your request. Could you provide more details about what you'd like to do with GitHub?"
                state["recommended_tools"] = []
                state["tool_details"] = []
                state["follow_up_questions"] = [
                    "What specific GitHub task are you trying to accomplish?",
                    "Are you working with issues, pull requests, repositories, or something else?",
                    "Would you like me to show you available tool categories?"
                ]
                return state
            
            # Format tools for LLM
            tools_context = self._format_tools_for_llm(search_results)
            
            # Create prompt for LLM
            prompt = f"""
USER REQUEST: {user_query}
DETECTED INTENT: {intent}

AVAILABLE TOOLS:
{tools_context}

Based on the user's request and intent, provide:
1. 1-3 most relevant tool recommendations (use exact tool names from above)
2. Clear reasoning for your recommendations
3. 2-3 follow-up questions to better assist the user

Focus on practical applicability and match the user's likely experience level.
Provide your response in a conversational format.
"""
            
            # Get LLM response
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response and update state
            parsed_response = self._parse_llm_response(response.content, search_results)
            
            state["reasoning"] = parsed_response["reasoning"]
            state["recommended_tools"] = parsed_response["recommended_tools"]
            state["tool_details"] = parsed_response["tools_details"]
            state["follow_up_questions"] = parsed_response["follow_up_questions"]
            state["current_step"] = "recommendations_generated"
            
        except Exception as e:
            state["error"] = str(e)
            logger.error(f"Recommendation generation failed: {e}")
        
        return state
    
    def _format_response(self, state: ChatState) -> ChatState:
        """Format the final response"""
        logger.info("Formatting response...")
        
        # Add the assistant response to messages
        reasoning = state.get("reasoning", "I'm having trouble processing your request.")
        
        assistant_message = AIMessage(content=reasoning)
        state["messages"].append(assistant_message)
        state["current_step"] = "response_formatted"
        
        return state
    
    def _handle_error(self, state: ChatState) -> ChatState:
        """Handle errors in the workflow"""
        logger.info("Handling error...")
        
        error_msg = state.get("error", "An unknown error occurred.")
        
        error_response = f"I encountered an error: {error_msg}\n\nPlease try rephrasing your question or type 'help' for assistance."
        
        assistant_message = AIMessage(content=error_response)
        state["messages"].append(assistant_message)
        
        return state
    
    def _check_for_errors(self, state: ChatState) -> str:
        """Check if there are errors in the current state"""
        return "error" if state.get("error") else "continue"
    
    # ==================== Helper Methods ====================
    
    def _format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool information for LLM processing"""
        formatted_tools = []
        for i, tool in enumerate(tools):
            formatted_tool = f"""
TOOL {i+1}:
Tool: {tool.get('toolName')}
Section: {tool.get('section')}
Description: {tool.get('description')}
Action Type: {tool.get('actionType')}
Resource Type: {tool.get('resourceType')}
Complexity: {tool.get('complexityLevel')}
Required Params: {', '.join(tool.get('requiredParams', []))}
Use Cases: {', '.join(tool.get('useCases', [])[:3])}
"""
            formatted_tools.append(formatted_tool)
        
        return "\n".join(formatted_tools)
    
    def _parse_llm_response(self, llm_response: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response and extract structured information"""
        try:
            # Create a mapping of tool names to tool details
            tools_map = {tool['toolName']: tool for tool in available_tools}
            
            # Extract recommended tools from LLM response
            recommended_tools = []
            tools_details = []
            
            for tool_name in tools_map.keys():
                if tool_name.lower() in llm_response.lower():
                    recommended_tools.append(tool_name)
                    tools_details.append(tools_map[tool_name])
            
            # If no specific tools mentioned, use top 3 from search results
            if not recommended_tools:
                recommended_tools = [tool['toolName'] for tool in available_tools[:3]]
                tools_details = available_tools[:3]
            
            return {
                "recommended_tools": recommended_tools,
                "reasoning": llm_response,
                "follow_up_questions": self._extract_questions_from_response(llm_response),
                "tools_details": tools_details
            }
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return {
                "recommended_tools": [tool['toolName'] for tool in available_tools[:2]],
                "reasoning": llm_response,
                "follow_up_questions": ["Would you like more details about these tools?"],
                "tools_details": available_tools[:2]
            }
    
    def _extract_questions_from_response(self, response: str) -> List[str]:
        """Extract follow-up questions from LLM response"""
        questions = []
        lines = response.split('\n')
        for line in lines:
            if '?' in line and len(line.strip()) > 10:
                questions.append(line.strip().strip('- '))
        
        # Default questions if none extracted
        if not questions:
            questions = [
                "Would you like more details about any of these tools?",
                "Do you need help with the required parameters?",
                "Would you like to see related tools?"
            ]
        
        return questions[:3]
    
    # ==================== Public Interface ====================
    
    def initialize_storage_if_needed(self):
        """Initialize storage system if not already done"""
        try:
            stats = self.storage.get_storage_stats()
            if stats.get('total_tools', 0) == 0:
                raise Exception("No tools found in storage")
            console.print(f"[green]✓ Storage ready with {stats['total_tools']} tools[/green]")
        except:
            console.print("[yellow]Storage not initialized. Initializing now...[/yellow]")
            self.storage.initialize_storage()
    
    def process_message(self, user_input: str, thread_id: str = "default") -> ChatState:
        """Process a user message through the LangGraph workflow"""
        
        # Create initial state
        initial_state = ChatState(
            messages=[HumanMessage(content=user_input)],
            user_query="",
            intent="",
            search_results=[],
            recommended_tools=[],
            tool_details=[],
            reasoning="",
            follow_up_questions=[],
            conversation_context={},
            current_step="",
            error=None
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.workflow.invoke(initial_state, config)
        
        return final_state
    
    def display_response(self, state: ChatState):
        """Display the chatbot response in a formatted way"""
        # Display main reasoning
        reasoning = state.get("reasoning", "No response generated.")
        console.print(Panel(
            Markdown(reasoning),
            title="💡 Analysis & Recommendations",
            border_style="blue"
        ))
        
        # Display recommended tools table
        tool_details = state.get("tool_details", [])
        if tool_details:
            table = Table(title="🔧 Recommended Tools")
            table.add_column("Tool", style="cyan", no_wrap=True)
            table.add_column("Section", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Complexity", style="green")
            table.add_column("Action", style="magenta")
            
            for tool in tool_details:
                table.add_row(
                    tool.get('toolName', ''),
                    tool.get('section', ''),
                    tool.get('description', '')[:60] + "..." if len(tool.get('description', '')) > 60 else tool.get('description', ''),
                    tool.get('complexityLevel', ''),
                    tool.get('actionType', '')
                )
            
            console.print(table)
        
        # Display follow-up questions
        follow_up_questions = state.get("follow_up_questions", [])
        if follow_up_questions:
            questions_text = "\n".join([f"• {q}" for q in follow_up_questions])
            console.print(Panel(
                questions_text,
                title="❓ Follow-up Questions",
                border_style="yellow"
            ))
    
    def display_tool_details(self, tool_name: str):
        """Display detailed information about a specific tool"""
        try:
            tool = self.storage.get_tool_by_name(tool_name)
            if not tool:
                console.print(f"[red]Tool '{tool_name}' not found[/red]")
                return
            
            # Create detailed display
            details = f"""
**Tool Name:** {tool.get('toolName')}
**Section:** {tool.get('section')}
**Full Name:** {tool.get('fullQualifiedName')}

**Description:** {tool.get('description')}

**Properties:**
• Action Type: {tool.get('actionType')}
• Resource Type: {tool.get('resourceType')}
• Complexity: {tool.get('complexityLevel')}
• Requires Mutation: {'Yes' if tool.get('isMutation') else 'No'}

**Parameters:**
• Required: {', '.join(tool.get('requiredParams', [])) or 'None'}
• Optional: {', '.join(tool.get('optionalParams', [])) or 'None'}

**Use Cases:**
{chr(10).join([f"• {case}" for case in tool.get('useCases', [])])}

**Related Tools:**
{', '.join(tool.get('relatedTools', [])) or 'None'}
"""
            
            console.print(Panel(
                Markdown(details),
                title=f"🔧 {tool_name} Details",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Error retrieving tool details: {e}[/red]")
    
    def run_interactive_session(self):
        """Run interactive chatbot session using LangGraph"""
        console.print(Panel.fit(
            "🤖 Welcome to LangGraph GitHub MCP Tools Chatbot!\n"
            "Ask me about GitHub tools and I'll help you find what you need.\n"
            "Type 'quit' to exit, 'help' for commands, or 'stats' for storage info.\n"
            "🆕 Now powered by LangGraph for better conversation handling!",
            style="bold blue"
        ))
        
        # Initialize storage
        self.initialize_storage_if_needed()
        
        thread_id = "interactive_session"
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print("[yellow]Goodbye! Happy coding! 👋[/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                elif user_input.lower().startswith('details '):
                    tool_name = user_input[8:].strip()
                    self.display_tool_details(tool_name)
                    continue
                
                elif user_input.lower() == 'clear':
                    # Create new thread ID to clear conversation
                    thread_id = f"interactive_session_{len(str(hash(user_input)))}"
                    console.print("[green]Conversation history cleared.[/green]")
                    continue
                
                # Process message through LangGraph workflow
                console.print("\n[yellow]🤔 Processing through LangGraph workflow...[/yellow]")
                
                final_state = self.process_message(user_input, thread_id)
                
                # Display the response
                self.display_response(final_state)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                console.print("[yellow]Please try again or type 'help' for assistance.[/yellow]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
**Available Commands:**
• `quit/exit/bye` - Exit the chatbot
• `help` - Show this help message
• `stats` - Show storage statistics
• `details <tool_name>` - Get detailed info about a specific tool
• `clear` - Clear conversation history (start new thread)

**Example Queries:**
• "How do I create a new issue?"
• "Show me tools for managing pull requests"
• "I need to search repositories"
• "What tools can help with GitHub workflows?"
• "How do I list all issues in a repository?"

**LangGraph Features:**
• Persistent conversation memory within sessions
• Structured workflow processing
• Better error handling and recovery
• Tool integration with automatic routing

**Tips:**
• Be specific about what you want to do with GitHub
• Mention the type of resource (issues, PRs, repos, etc.)
• Ask for tools by complexity level if needed
• The bot remembers context within your session
"""
        console.print(Panel(
            Markdown(help_text),
            title="❓ Help & Usage Guide",
            border_style="yellow"
        ))
    
    def _show_stats(self):
        """Show storage statistics"""
        try:
            stats = self.storage.get_storage_stats()
            
            stats_text = f"""
**Storage Statistics:**
• Total Tools: {stats.get('total_tools', 0)}
• Collection: {stats.get('collection_name', 'Unknown')}

**Tools by Section:**
"""
            for section, count in stats.get('tools_by_section', {}).items():
                stats_text += f"• {section}: {count} tools\n"
            
            stats_text += "\n**Tools by Action Type:**\n"
            for action, count in stats.get('tools_by_action', {}).items():
                stats_text += f"• {action}: {count} tools\n"
            
            stats_text += f"""

**LangGraph Workflow Status:**
• Nodes: Input Processor → Intent Analyzer → Tool Searcher → Recommendation Generator → Response Formatter
• Memory: MemorySaver (conversation persistence)
• Tools: {len(self.tools)} available tools
• Error Handling: Built-in error recovery
"""
            
            console.print(Panel(
                Markdown(stats_text),
                title="📊 Storage & Workflow Statistics",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Failed to get stats: {e}[/red]")

def main():
    """Main function to run the LangGraph chatbot"""
    # Configuration
    JSON_FILE_PATH = "github_mcp_tools.json"  # Update with your actual path
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    try:
        # Initialize LangGraph chatbot
        chatbot = LangGraphGitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY
        )
        
        # Run interactive session
        chatbot.run_interactive_session()
        
    except Exception as e:
        console.print(f"[red]Failed to initialize LangGraph chatbot: {e}[/red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")

if __name__ == "__main__":
    main()