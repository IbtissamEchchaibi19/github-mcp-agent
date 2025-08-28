import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# Import your existing storage system
from Weaviate import EnhancedGitHubMCPStorage
# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

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

class GitHubMCPChatbot:
    """Intelligent chatbot for GitHub MCP Tools"""
    
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
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ChatbotResponse)
        
        # System prompt for the LLM
        self.system_prompt = self._create_system_prompt()
        
        console.print(Panel.fit("🤖 GitHub MCP Tools Chatbot Initialized", style="bold blue"))
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize the Groq LLM"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        logger.info(f"API Key loaded: {'Yes' if api_key else 'No'}")
        
        return ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Using a more stable model
            temperature=0.1,  # Lower temperature for more consistent tool selection
            max_completion_tokens=1024,
            top_p=0.9,
            stream=False,
            stop=None,
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

ANALYSIS APPROACH:
1. Extract the primary action (create, read, update, delete, list, search)
2. Identify the GitHub resource type (issue, PR, repo, workflow, etc.)
3. Consider the user's experience level and complexity preference
4. Recommend tools that best match the intent and context

Remember: Focus on practical tool recommendations that directly address the user's GitHub workflow needs."""

    def initialize_storage_if_needed(self):
        """Initialize storage system if not already done"""
        try:
            # Try to get stats to see if storage is initialized
            stats = self.storage.get_storage_stats()
            if stats.get('total_tools', 0) == 0:
                raise Exception("No tools found in storage")
            console.print(f"[green]✓ Storage ready with {stats['total_tools']} tools[/green]")
        except:
            console.print("[yellow]Storage not initialized. Initializing now...[/yellow]")
            self.storage.initialize_storage()
    
    def search_relevant_tools(self, user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant tools based on user query"""
        try:
            results = self.storage.search_tools(user_query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Tool search failed: {e}")
            return []
    
    def format_tool_for_llm(self, tool: Dict[str, Any]) -> str:
        """Format tool information for LLM processing"""
        return f"""
Tool: {tool.get('toolName')}
Section: {tool.get('section')}
Description: {tool.get('description')}
Action Type: {tool.get('actionType')}
Resource Type: {tool.get('resourceType')}
Complexity: {tool.get('complexityLevel')}
Required Params: {', '.join(tool.get('requiredParams', []))}
Use Cases: {', '.join(tool.get('useCases', [])[:3])}
"""
    
    def generate_response(self, user_query: str) -> Dict[str, Any]:
        """Generate chatbot response with tool recommendations"""
        try:
            # Search for relevant tools
            relevant_tools = self.search_relevant_tools(user_query, top_k=8)
            
            if not relevant_tools:
                return {
                    "intent": "unclear",
                    "recommended_tools": [],
                    "reasoning": "I couldn't find specific tools matching your request. Could you provide more details about what you'd like to do with GitHub?",
                    "follow_up_questions": [
                        "What specific GitHub task are you trying to accomplish?",
                        "Are you working with issues, pull requests, repositories, or something else?",
                        "Would you like me to show you available tool categories?"
                    ],
                    "tools_details": []
                }
            
            # Format tools for LLM
            tools_context = "\n".join([
                f"TOOL {i+1}:" + self.format_tool_for_llm(tool) 
                for i, tool in enumerate(relevant_tools)
            ])
            
            # Create prompt for LLM
            prompt = f"""
USER REQUEST: {user_query}

AVAILABLE TOOLS:
{tools_context}

Based on the user's request and the available tools above, provide:
1. The detected intent
2. 1-3 most relevant tool recommendations (use exact tool names from above)
3. Clear reasoning for your recommendations
4. 2-3 follow-up questions to better assist the user

Focus on practical applicability and match the user's likely experience level.
"""
            
            # Get LLM response
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response
            return self._parse_llm_response(response.content, relevant_tools)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "intent": "error",
                "recommended_tools": [],
                "reasoning": f"I encountered an error processing your request: {str(e)}",
                "follow_up_questions": ["Could you try rephrasing your question?"],
                "tools_details": []
            }
    
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
                "intent": self._extract_intent_from_response(llm_response),
                "recommended_tools": recommended_tools,
                "reasoning": llm_response,
                "follow_up_questions": self._extract_questions_from_response(llm_response),
                "tools_details": tools_details
            }
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return {
                "intent": "unknown",
                "recommended_tools": [tool['toolName'] for tool in available_tools[:2]],
                "reasoning": llm_response,
                "follow_up_questions": ["Would you like more details about these tools?"],
                "tools_details": available_tools[:2]
            }
    
    def _extract_intent_from_response(self, response: str) -> str:
        """Extract user intent from LLM response"""
        response_lower = response.lower()
        if any(word in response_lower for word in ['create', 'make', 'new']):
            return 'create'
        elif any(word in response_lower for word in ['list', 'show', 'view', 'get']):
            return 'list'
        elif any(word in response_lower for word in ['update', 'edit', 'modify']):
            return 'update'
        elif any(word in response_lower for word in ['search', 'find']):
            return 'search'
        else:
            return 'general'
    
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
    
    def display_response(self, response: Dict[str, Any]):
        """Display the chatbot response in a formatted way"""
        # Display main reasoning
        console.print(Panel(
            Markdown(response['reasoning']),
            title="💡 Analysis & Recommendations",
            border_style="blue"
        ))
        
        # Display recommended tools table
        if response['tools_details']:
            table = Table(title="🔧 Recommended Tools")
            table.add_column("Tool", style="cyan", no_wrap=True)
            table.add_column("Section", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Complexity", style="green")
            table.add_column("Action", style="magenta")
            
            for tool in response['tools_details']:
                table.add_row(
                    tool.get('toolName', ''),
                    tool.get('section', ''),
                    tool.get('description', '')[:60] + "..." if len(tool.get('description', '')) > 60 else tool.get('description', ''),
                    tool.get('complexityLevel', ''),
                    tool.get('actionType', '')
                )
            
            console.print(table)
        
        # Display follow-up questions
        if response['follow_up_questions']:
            questions_text = "\n".join([f"• {q}" for q in response['follow_up_questions']])
            console.print(Panel(
                questions_text,
                title="❓ Follow-up Questions",
                border_style="yellow"
            ))
    
    def get_tool_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        try:
            return self.storage.get_tool_by_name(tool_name)
        except Exception as e:
            logger.error(f"Failed to get tool details: {e}")
            return None
    
    def display_tool_details(self, tool_name: str):
        """Display detailed information about a specific tool"""
        tool = self.get_tool_details(tool_name)
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
    
    def run_interactive_session(self):
        """Run interactive chatbot session"""
        console.print(Panel.fit(
            "🤖 Welcome to GitHub MCP Tools Chatbot!\n"
            "Ask me about GitHub tools and I'll help you find what you need.\n"
            "Type 'quit' to exit, 'help' for commands, or 'stats' for storage info.",
            style="bold blue"
        ))
        
        # Initialize storage
        self.initialize_storage_if_needed()
        
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
                    self.conversation_history = []
                    console.print("[green]Conversation history cleared.[/green]")
                    continue
                
                # Generate and display response
                console.print("\n[yellow]🤔 Analyzing your request...[/yellow]")
                response = self.generate_response(user_input)
                self.display_response(response)
                
                # Add to conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'response': response
                })
                
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
• `clear` - Clear conversation history

**Example Queries:**
• "How do I create a new issue?"
• "Show me tools for managing pull requests"
• "I need to search repositories"
• "What tools can help with GitHub workflows?"
• "How do I list all issues in a repository?"

**Tips:**
• Be specific about what you want to do with GitHub
• Mention the type of resource (issues, PRs, repos, etc.)
• Ask for tools by complexity level if needed
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
            
            console.print(Panel(
                Markdown(stats_text),
                title="📊 Storage Statistics",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Failed to get stats: {e}[/red]")

def main():
    """Main function to run the chatbot"""
    # Configuration
    JSON_FILE_PATH = "github_mcp_tools.json"  # Update with your actual path
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    try:
        # Initialize chatbot
        chatbot = GitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY
        )
        
        # Run interactive session
        chatbot.run_interactive_session()
        
    except Exception as e:
        console.print(f"[red]Failed to initialize chatbot: {e}[/red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")

if __name__ == "__main__":
    main()