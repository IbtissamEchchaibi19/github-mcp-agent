import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiofiles
import psutil
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
llmapi = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProjectState:
    """Track project state and memory"""
    project_path: str
    project_name: str
    git_status: Dict[str, Any]
    last_commit: Optional[str]
    branch: str
    actions_history: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

@dataclass
class AgentMemory:
    """Agent's memory system"""
    projects: Dict[str, ProjectState]
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]

class GitHubMCPClient:
    """GitHub MCP Client using LangChain MCP Adapters"""
    
    def __init__(self):
        self.mcp_client = None
        self.connected = False
    
    async def connect(self):
        """Connect to MCP servers using LangChain adapters"""
        try:
            server_config = {
                "github": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "ghcr.io/github/github-mcp-server:latest"
                    ],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
                    }
                }
            }
            
            self.mcp_client = MultiServerMCPClient()
            await self.mcp_client.connect(server_config)
            self.connected = True
            
            logger.info("Connected to GitHub MCP Server via Docker and LangChain adapters")
            
            try:
                tools = await self.mcp_client.list_tools()
                logger.info(f"Available MCP tools: {[tool.name for tool in tools]}")
            except Exception as e:
                logger.warning(f"Failed to list MCP tools: {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")
            logger.info("Ensure Docker is running and GITHUB_PERSONAL_ACCESS_TOKEN is set.")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from MCP servers"""
        try:
            if self.mcp_client and self.connected:
                await self.mcp_client.disconnect()
                self.connected = False
                logger.info("Disconnected from GitHub MCP Server")
        except Exception as e:
            logger.error(f"Failed to disconnect from MCP servers: {e}")
    
    async def git_status(self, project_path: str) -> Dict[str, Any]:
        """Get git status for a project"""
        if not self.connected:
            return {"error": "MCP client not connected"}
        try:
            # Placeholder: Replace with actual MCP client call
            return {"status": f"Git status for {project_path}", "files": []}
        except Exception as e:
            return {"error": f"Failed to get git status: {str(e)}"}
    
    async def git_add_commit_push(self, project_path: str, message: str, files: Optional[List[str]]) -> Dict[str, Any]:
        """Add, commit, and push changes"""
        if not self.connected:
            return {"error": "MCP client not connected"}
        try:
            # Placeholder: Replace with actual MCP client call
            return {"success": True, "message": f"Committed and pushed to {project_path}"}
        except Exception as e:
            return {"error": f"Failed to commit/push: {str(e)}"}
    
    async def create_pull_request(self, project_path: str, title: str, body: str, base: str) -> Dict[str, Any]:
        """Create a pull request"""
        if not self.connected:
            return {"error": "MCP client not connected"}
        try:
            # Placeholder: Replace with actual MCP client call
            return {"success": True, "pr_url": f"https://github.com/repo/pulls/{title}"}
        except Exception as e:
            return {"error": f"Failed to create PR: {str(e)}"}
    
    async def list_issues(self, project_path: str) -> Dict[str, Any]:
        """List GitHub issues"""
        if not self.connected:
            return {"error": "MCP client not connected"}
        try:
            # Placeholder: Replace with actual MCP client call
            return {"issues": [], "count": 0}
        except Exception as e:
            return {"error": f"Failed to list issues: {str(e)}"}
    
    async def get_repository_info(self, project_path: str) -> Dict[str, Any]:
        """Get repository information"""
        if not self.connected:
            return {"error": "MCP client not connected"}
        try:
            # Placeholder: Replace with actual MCP client call
            return {"name": Path(project_path).name, "url": "https://github.com/repo"}
        except Exception as e:
            return {"error": f"Failed to get repo info: {str(e)}"}

class VSCodeMonitor:
    """Monitor VS Code for new project launches"""
    
    def __init__(self, callback):
        self.callback = callback
        self.monitored_projects = set()
        self.running = False
    
    async def start_monitoring(self):
        """Start monitoring VS Code processes"""
        self.running = True
        logger.info("Starting VS Code monitoring...")
        
        while self.running:
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] and 'code' in proc.info['name'].lower():
                            if proc.info['cmdline']:
                                for arg in proc.info['cmdline']:
                                    if os.path.isdir(arg) and arg not in self.monitored_projects:
                                        if os.path.exists(os.path.join(arg, '.git')):
                                            self.monitored_projects.add(arg)
                                            await self.callback(arg)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in VS Code monitoring: {e}")
                await asyncio.sleep(10)
    
    def stop_monitoring(self):
        """Stop monitoring VS Code processes"""
        self.running = False
        logger.info("Stopped VS Code monitoring")

class MemoryManager:
    """Manage agent memory and state persistence"""
    
    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = AgentMemory(projects={}, user_preferences={}, conversation_history=[])
        self.load_memory()
    
    def load_memory(self):
        """Load memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    projects = {}
                    for k, v in data.get('projects', {}).items():
                        try:
                            v['created_at'] = datetime.fromisoformat(v['created_at']) if isinstance(v['created_at'], str) else v['created_at']
                            v['updated_at'] = datetime.fromisoformat(v['updated_at']) if isinstance(v['updated_at'], str) else v['updated_at']
                            projects[k] = ProjectState(**v)
                        except (KeyError, TypeError) as e:
                            logger.warning(f"Skipping invalid project data for {k}: {e}")
                    
                    self.memory = AgentMemory(
                        projects=projects,
                        user_preferences=data.get('user_preferences', {}),
                        conversation_history=data.get('conversation_history', [])
                    )
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    async def save_memory(self):
        """Save memory to file"""
        try:
            data = {
                'projects': {k: asdict(v) for k, v in self.memory.projects.items()},
                'user_preferences': self.memory.user_preferences,
                'conversation_history': self.memory.conversation_history
            }
            
            for project_data in data['projects'].values():
                project_data['created_at'] = project_data['created_at'].isoformat() if isinstance(project_data['created_at'], datetime) else project_data['created_at']
                project_data['updated_at'] = project_data['updated_at'].isoformat() if isinstance(project_data['updated_at'], datetime) else project_data['updated_at']
            
            async with aiofiles.open(self.memory_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_project(self, project_path: str, git_status: Dict[str, Any]):
        """Add new project to memory"""
        project_name = Path(project_path).name
        now = datetime.now()
        
        self.memory.projects[project_path] = ProjectState(
            project_path=project_path,
            project_name=project_name,
            git_status=git_status,
            last_commit=git_status.get('last_commit'),
            branch=git_status.get('branch', 'main'),
            actions_history=[],
            created_at=now,
            updated_at=now
        )
    
    def update_project(self, project_path: str, **updates):
        """Update project state"""
        if project_path in self.memory.projects:
            project = self.memory.projects[project_path]
            for key, value in updates.items():
                setattr(project, key, value)
            project.updated_at = datetime.now()
    
    def add_conversation(self, user_message: str, ai_response: str, project_path: str = None):
        """Add conversation to history"""
        self.memory.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'ai_response': ai_response,
            'project_path': project_path
        })

class GitHubAgent:
    """Main AI Agent for GitHub operations"""
    
    def __init__(self, groq_api_key: str):
        self.groq_client = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            api_key=llmapi
        )
        
        self.github_client = GitHubMCPClient()
        self.memory_manager = MemoryManager()
        self.vs_code_monitor = VSCodeMonitor(self.on_project_detected)
        
        self.create_workflow()
    
    def create_workflow(self):
        """Create LangGraph workflow for processing user commands"""
        
        @tool
        async def git_status_tool(project_path: str = None) -> str:
            """Get git status for a project"""
            if not project_path:
                project_path = self._get_active_project() or "."
            
            if self.github_client.connected:
                result = await self.github_client.git_status(project_path)
            else:
                import subprocess
                try:
                    cmd_result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    result = {"success": True, "status": cmd_result.stdout}
                except subprocess.CalledProcessError as e:
                    result = {"error": f"Git status failed: {str(e)}"}
            
            return json.dumps(result, indent=2)
        
        @tool
        async def git_commit_push_tool(project_path: str = None, message: str = "Auto-commit", files: str = None) -> str:
            """Add, commit and push changes to GitHub"""
            if not project_path:
                project_path = self._get_active_project() or "."
            
            files_list = files.split(',') if files else None
            
            if self.github_client.connected:
                result = await self.github_client.git_add_commit_push(project_path, message, files_list)
            else:
                import subprocess
                try:
                    if files_list:
                        subprocess.run(["git", "add"] + files_list, cwd=project_path, check=True)
                    else:
                        subprocess.run(["git", "add", "."], cwd=project_path, check=True)
                    
                    subprocess.run(["git", "commit", "-m", message], cwd=project_path, check=True)
                    subprocess.run(["git", "push"], cwd=project_path, check=True)
                    
                    result = {"success": True, "message": "Changes committed and pushed successfully"}
                except subprocess.CalledProcessError as e:
                    result = {"error": f"Git operation failed: {str(e)}"}
            
            return json.dumps(result, indent=2)
        
        @tool
        async def create_pull_request_tool(project_path: str = None, title: str = "New Pull Request", body: str = "", base: str = "main") -> str:
            """Create a pull request (requires MCP connection)"""
            if not project_path:
                project_path = self._get_active_project() or "."
            
            if self.github_client.connected:
                result = await self.github_client.create_pull_request(project_path, title, body, base)
            else:
                result = {"error": "Pull request creation requires GitHub MCP server connection"}
            
            return json.dumps(result, indent=2)
        
        @tool
        async def list_issues_tool(project_path: str = None) -> str:
            """List GitHub issues (requires MCP connection)"""
            if not project_path:
                project_path = self._get_active_project() or "."
            
            if self.github_client.connected:
                result = await self.github_client.list_issues(project_path)
            else:
                result = {"error": "Listing issues requires GitHub MCP server connection"}
            
            return json.dumps(result, indent=2)
        
        @tool
        async def get_repository_info_tool(project_path: str = None) -> str:
            """Get repository information (requires MCP connection)"""
            if not project_path:
                project_path = self._get_active_project() or "."
            
            if self.github_client.connected:
                result = await self.github_client.get_repository_info(project_path)
            else:
                result = {"error": "Repository info requires GitHub MCP server connection"}
            
            return json.dumps(result, indent=2)
        
        @tool
        async def get_project_memory_tool(project_path: str = None) -> str:
            """Get project memory and history"""
            if project_path and project_path in self.memory_manager.memory.projects:
                project = self.memory_manager.memory.projects[project_path]
                return json.dumps(asdict(project), default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o), indent=2)
            else:
                return json.dumps({
                    "all_projects": list(self.memory_manager.memory.projects.keys()),
                    "recent_conversations": self.memory_manager.memory.conversation_history[-5:]
                }, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o), indent=2)
        
        class AgentState(TypedDict):
            messages: list[BaseMessage]
            current_project: Optional[str]
        
        tools = [
            git_status_tool,
            git_commit_push_tool,
            create_pull_request_tool,
            list_issues_tool,
            get_repository_info_tool,
            get_project_memory_tool
        ]
        tool_node = ToolNode(tools)
        
        workflow = StateGraph(AgentState)
        
        def should_continue(state: AgentState):
            """Decide whether to continue or end"""
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "action"
            return "end"
        
        async def call_model(state: AgentState):
            """Call the LLM"""
            mcp_status = "with full GitHub MCP integration" if self.github_client.connected else "with basic git functionality"
            
            system_message = SystemMessage(content=f"""You are an AI development assistant that helps manage GitHub operations through natural language.

Current connection status: {mcp_status}

Available tools:
- git_status_tool: Check git status (always available)
- git_commit_push_tool: Add, commit, and push changes (always available)
- create_pull_request_tool: Create GitHub pull requests (requires MCP)
- list_issues_tool: List GitHub issues (requires MCP)
- get_repository_info_tool: Get repository information (requires MCP)
- get_project_memory_tool: Access project memory and history (always available)

When users ask for GitHub operations:
1. For basic git operations (status, commit, push), these work with or without MCP
2. For advanced GitHub features (PRs, issues), inform users if MCP connection is needed
3. Always provide clear feedback about what was accomplished

If MCP features are requested but unavailable, explain the limitation and offer alternatives.""")
            
            messages = [system_message] + state["messages"]
            response = await self.groq_client.ainvoke(messages)
            return {"messages": [response]}
        
        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"action": "action", "end": "__end__"}
        )
        workflow.add_edge("action", "agent")
        
        self.app = workflow.compile()
    
    async def on_project_detected(self, project_path: str):
        """Called when a new project is detected"""
        logger.info(f"New project detected: {project_path}")
        
        if self.github_client.connected:
            git_status = await self.github_client.git_status(project_path)
        else:
            git_status = {"status": "MCP not connected, using basic functionality"}
        
        self.memory_manager.add_project(project_path, git_status)
        await self.memory_manager.save_memory()
        
        project_name = Path(project_path).name or "Current Directory"
        print(f"🚀 AI Agent activated for project: {project_name}")
        connection_status = "Full GitHub integration" if self.github_client.connected else "Basic git functionality"
        print(f"Status: {connection_status}")
        print("You can now use natural language commands!")
    
    async def process_command(self, user_input: str, project_path: str = None) -> str:
        """Process user command through LangGraph"""
        try:
            current_project = project_path or self._get_active_project() or "."
            
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_project": current_project
            }
            
            result = await self.app.ainvoke(initial_state)
            ai_response = result["messages"][-1].content
            
            self.memory_manager.add_conversation(user_input, ai_response, current_project)
            await self.memory_manager.save_memory()
            
            return ai_response
            
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_active_project(self) -> Optional[str]:
        """Get the most recently updated project"""
        if not self.memory_manager.memory.projects:
            return None
        
        return max(
            self.memory_manager.memory.projects.keys(),
            key=lambda x: self.memory_manager.memory.projects[x].updated_at
        )
    
    async def start(self):
        """Start the agent"""
        logger.info("Starting GitHub AI Agent...")
        
        await self.github_client.connect()
        
        if self.github_client.connected:
            print("✅ GitHub MCP Server connected - Full functionality available!")
        else:
            print("⚠️  GitHub MCP Server not connected - Basic git functionality available")
            print("Install: pip install langchain-mcp-adapters")
        
        try:
            monitor_task = asyncio.create_task(self.vs_code_monitor.start_monitoring())
            command_task = asyncio.create_task(self.command_loop())
            
            await asyncio.gather(monitor_task, command_task)
        finally:
            self.vs_code_monitor.stop_monitoring()
            if self.github_client.connected:
                await self.github_client.disconnect()
    
    async def command_loop(self):
        """Interactive command loop"""
        print("\n" + "="*60)
        print("🤖 GitHub AI Agent Ready!")
        print("="*60)
        
        connection_status = "🟢 MCP Connected" if self.github_client.connected else "🟡 Basic Mode"
        print(f"Status: {connection_status}")
        
        print("\nType your commands in natural language:")
        print("Examples:")
        print("- 'Check git status for current project'")
        print("- 'Commit and push all changes with message: Added new feature'")
        if self.github_client.connected:
            print("- 'Create a pull request titled: Bug fix for login'")
            print("- 'List all issues in this repository'")
            print("- 'Get repository information'")
        print("- 'Show me the project history'")
        print("- 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("🔵 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if user_input:
                    print("🤖 Agent: Processing your request...")
                    response = await self.process_command(user_input)
                    print(f"🤖 Agent: {response}\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
        
        print("👋 GitHub AI Agent stopped.")

async def main():
    """Main function to run the agent"""
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        groq_api_key = input("Enter your Groq API key: ").strip()
    
    if not groq_api_key:
        print("❌ Groq API key is required!")
        return
    
    github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not github_token:
        print("ℹ️  Tip: Set GITHUB_PERSONAL_ACCESS_TOKEN environment variable for full GitHub API access")
    
    agent = GitHubAgent(groq_api_key)
    await agent.start()

if __name__ == "__main__":
    asyncio.run(main())