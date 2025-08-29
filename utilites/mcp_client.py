import json
import subprocess
import logging
import time
import argparse
import sys
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import os
import re
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class GitHubToolRequest(BaseModel):
    """Schema for GitHub tool requests"""
    tool_name: str = Field(description="Name of the GitHub MCP tool to invoke")
    parameters: Dict[str, Any] = Field(description="Parameters to pass to the tool")

class AgentState(TypedDict):
    """State of the agent"""
    messages: Annotated[List[Any], add_messages]
    tool_request: Optional[GitHubToolRequest]
    mcp_response: Optional[Dict[str, Any]]
    error: Optional[str]

class GitHubMCPAgent:
    def __init__(self, groq_api_key: str):
        self.process = None
        self.request_id = 1
        
        # Initialize ChatGroq LLM
        self.llm = ChatGroq(
            model="meta-llama/llama-3.3-70b-versatile",  # Updated to a more reliable model
            temperature=0,
            max_completion_tokens=200,
            top_p=0.9,
            stream=False,
            api_key=groq_api_key
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_request", self._parse_request)
        workflow.add_node("connect_mcp", self._connect_mcp)
        workflow.add_node("invoke_tool", self._invoke_tool)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("parse_request")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "parse_request",
            self._check_parse_result,
            {"success": "connect_mcp", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "connect_mcp",
            self._check_connection,
            {"success": "invoke_tool", "error": "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "invoke_tool",
            self._check_tool_result,
            {"success": "format_response", "error": "handle_error"}
        )
        
        # End nodes
        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _check_parse_result(self, state: AgentState) -> str:
        """Check if parsing was successful"""
        return "error" if state.get("error") or not state.get("tool_request") else "success"
    
    def _check_connection(self, state: AgentState) -> str:
        """Check if MCP connection was successful"""
        return "error" if state.get("error") else "success"
    
    def _check_tool_result(self, state: AgentState) -> str:
        """Check if tool invocation was successful"""
        return "error" if state.get("error") else "success"
    
    def _parse_request(self, state: AgentState) -> AgentState:
        """Parse the user request to extract tool name and parameters"""
        try:
            user_message = state["messages"][-1].content
            
            # Try to extract JSON from the message (for direct tool calls)
            json_match = re.search(r'\{.*\}', user_message, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    parsed_data = json.loads(json_str)
                    if "tool_name" in parsed_data and "parameters" in parsed_data:
                        state["tool_request"] = GitHubToolRequest(**parsed_data)
                        logger.info(f"Parsed tool request from JSON: {state['tool_request']}")
                        return state
                except json.JSONDecodeError:
                    pass
            
            # Fallback to LLM parsing with comprehensive tool definitions
            system_prompt = """You are a GitHub tool request parser. Extract tool name and parameters from natural language.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "tool_name": "exact_tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}

Available GitHub MCP Tools (grouped by category):

REPOSITORIES:
- create_branch: {"owner": "string", "repo": "string", "branch": "string", "from_branch": "string"}
- list_branches: {"owner": "string", "repo": "string"}
- get_file_contents: {"owner": "string", "repo": "string", "path": "string", "ref": "string"}
- create_or_update_file: {"owner": "string", "repo": "string", "path": "string", "content": "string", "message": "string", "branch": "string", "sha": "string"}
- delete_file: {"owner": "string", "repo": "string", "path": "string", "message": "string", "branch": "string"}
- list_commits: {"owner": "string", "repo": "string", "sha": "string", "author": "string"}
- get_commit: {"owner": "string", "repo": "string", "sha": "string"}
- list_tags: {"owner": "string", "repo": "string"}
- get_tag: {"owner": "string", "repo": "string", "tag": "string"}
- search_repositories: {"query": "string"}
- search_code: {"q": "string"}
- create_repository: {"name": "string", "description": "string", "private": "boolean"}
- fork_repository: {"owner": "string", "repo": "string", "organization": "string"}
- push_files: {"owner": "string", "repo": "string", "branch": "string", "files": "array", "message": "string"}

ISSUES:
- create_issue: {"owner": "string", "repo": "string", "title": "string", "body": "string", "assignees": "array", "labels": "array"}
- get_issue: {"owner": "string", "repo": "string", "issue_number": "number"}
- list_issues: {"owner": "string", "repo": "string", "state": "string", "labels": "array"}
- update_issue: {"owner": "string", "repo": "string", "issue_number": "number", "title": "string", "body": "string", "state": "string"}
- add_issue_comment: {"owner": "string", "repo": "string", "issue_number": "number", "body": "string"}
- get_issue_comments: {"owner": "string", "repo": "string", "issue_number": "number"}
- search_issues: {"query": "string"}
- assign_copilot_to_issue: {"owner": "string", "repo": "string", "issueNumber": "number"}

PULL REQUESTS:
- create_pull_request: {"owner": "string", "repo": "string", "title": "string", "body": "string", "head": "string", "base": "string", "draft": "boolean"}
- get_pull_request: {"owner": "string", "repo": "string", "pullNumber": "number"}
- list_pull_requests: {"owner": "string", "repo": "string", "state": "string", "base": "string"}
- update_pull_request: {"owner": "string", "repo": "string", "pullNumber": "number", "title": "string", "body": "string", "state": "string"}
- merge_pull_request: {"owner": "string", "repo": "string", "pullNumber": "number", "merge_method": "string"}
- get_pull_request_diff: {"owner": "string", "repo": "string", "pullNumber": "number"}
- get_pull_request_files: {"owner": "string", "repo": "string", "pullNumber": "number"}
- get_pull_request_comments: {"owner": "string", "repo": "string", "pullNumber": "number"}
- get_pull_request_reviews: {"owner": "string", "repo": "string", "pullNumber": "number"}
- get_pull_request_status: {"owner": "string", "repo": "string", "pullNumber": "number"}
- search_pull_requests: {"query": "string"}
- request_copilot_review: {"owner": "string", "repo": "string", "pullNumber": "number"}

ACTIONS (CI/CD):
- list_workflows: {"owner": "string", "repo": "string"}
- list_workflow_runs: {"owner": "string", "repo": "string", "workflow_id": "string"}
- get_workflow_run: {"owner": "string", "repo": "string", "run_id": "number"}
- cancel_workflow_run: {"owner": "string", "repo": "string", "run_id": "number"}
- rerun_workflow_run: {"owner": "string", "repo": "string", "run_id": "number"}
- run_workflow: {"owner": "string", "repo": "string", "workflow_id": "string", "ref": "string", "inputs": "object"}
- list_workflow_jobs: {"owner": "string", "repo": "string", "run_id": "number"}
- get_job_logs: {"owner": "string", "repo": "string", "job_id": "number", "run_id": "number"}

SECURITY:
- list_code_scanning_alerts: {"owner": "string", "repo": "string", "state": "string", "severity": "string"}
- get_code_scanning_alert: {"owner": "string", "repo": "string", "alertNumber": "number"}
- list_dependabot_alerts: {"owner": "string", "repo": "string", "state": "string", "severity": "string"}
- get_dependabot_alert: {"owner": "string", "repo": "string", "alertNumber": "number"}
- list_secret_scanning_alerts: {"owner": "string", "repo": "string", "state": "string"}
- get_secret_scanning_alert: {"owner": "string", "repo": "string", "alertNumber": "number"}

NOTIFICATIONS:
- list_notifications: {"owner": "string", "repo": "string", "filter": "string"}
- get_notification_details: {"notificationID": "string"}
- dismiss_notification: {"threadID": "string", "state": "string"}
- mark_all_notifications_read: {"owner": "string", "repo": "string"}

DISCUSSIONS:
- list_discussions: {"owner": "string", "repo": "string", "category": "string"}
- get_discussion: {"owner": "string", "repo": "string", "discussionNumber": "number"}
- get_discussion_comments: {"owner": "string", "repo": "string", "discussionNumber": "number"}
- list_discussion_categories: {"owner": "string", "repo": "string"}

CONTEXT & USERS:
- get_me: {}
- search_users: {"query": "string"}
- search_orgs: {"query": "string"}

Parse repository format "owner/repo" into separate owner and repo parameters.
Extract all relevant parameters from the user's request."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            response_content = response.content.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                state["tool_request"] = GitHubToolRequest(**parsed_data)
                logger.info(f"Parsed tool request with LLM: {state['tool_request']}")
            else:
                state["error"] = f"Could not extract JSON from LLM response: {response_content}"
                logger.error(state["error"])
            
        except json.JSONDecodeError as e:
            state["error"] = f"Invalid JSON in LLM response: {e}"
            logger.error(state["error"])
        except Exception as e:
            state["error"] = f"Error parsing request: {str(e)}"
            logger.error(state["error"])
        
        return state
    
    def _connect_mcp(self, state: AgentState) -> AgentState:
        """Connect to the MCP server"""
        try:
            logger.info("Connecting to GitHub MCP server...")
            
            # Make sure Docker MCP is available
            self.process = subprocess.Popen(
                ['docker', 'run', '-i', '--rm', '-e', 'GITHUB_PERSONAL_ACCESS_TOKEN', 'ghcr.io/github/github-mcp-server'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env={**os.environ, 'GITHUB_PERSONAL_ACCESS_TOKEN': os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN', '')}
            )
            
            time.sleep(3)
            
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                state["error"] = f"GitHub MCP server failed to start: {stderr_output}"
                logger.error(state["error"])
                return state
            
            # Initialize the server
            if not self._initialize_server():
                state["error"] = "Failed to initialize GitHub MCP server"
                return state
            
            logger.info("Connected to GitHub MCP server successfully")
            
        except Exception as e:
            state["error"] = f"Failed to connect to GitHub MCP server: {str(e)}"
            logger.error(state["error"])
        
        return state
    
    def _initialize_server(self) -> bool:
        """Initialize the MCP server"""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "github-mcp-agent",
                "version": "1.0.0"
            }
        }
        
        response = self._send_mcp_request("initialize", params)
        if response and "result" in response:
            # Send initialized notification
            self._send_mcp_request("initialized", {})
            return True
        return False
    
    def _send_mcp_request(self, method: str, params=None) -> Optional[Dict]:
        """Send a JSON-RPC request to the MCP server"""
        if not self.process:
            return None
        
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
            logger.debug(f"Sending MCP request: {request_json.strip()}")
            
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                return None
            
            response_data = json.loads(response_line.strip())
            logger.debug(f"Received MCP response: {response_data}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            return None
    
    def _invoke_tool(self, state: AgentState) -> AgentState:
        """Invoke the requested GitHub tool"""
        try:
            tool_request = state["tool_request"]
            
            logger.info(f"Invoking GitHub tool: {tool_request.tool_name}")
            logger.info(f"Parameters: {tool_request.parameters}")
            
            params = {
                "name": tool_request.tool_name,
                "arguments": tool_request.parameters
            }
            
            response = self._send_mcp_request("tools/call", params)
            
            if response and "result" in response:
                state["mcp_response"] = response["result"]
                logger.info("GitHub tool invoked successfully!")
            elif response and "error" in response:
                error = response["error"]
                state["error"] = f"GitHub MCP tool error: {error.get('message', 'Unknown error')} (Code: {error.get('code', 'N/A')})"
                logger.error(state["error"])
            else:
                state["error"] = f"Unexpected GitHub MCP response: {response}"
                logger.error(state["error"])
            
        except Exception as e:
            state["error"] = f"Error invoking GitHub tool: {str(e)}"
            logger.error(state["error"])
        
        return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """Format the final response"""
        try:
            mcp_response = state["mcp_response"]
            tool_request = state["tool_request"]
            
            # Format the response based on content type
            if "content" in mcp_response:
                content_text = ""
                for content_item in mcp_response["content"]:
                    if content_item.get("type") == "text":
                        content_text += content_item.get("text", "")
                
                response_message = f"✅ Successfully executed '{tool_request.tool_name}'\n\n{content_text}"
            else:
                # For structured responses, format nicely
                formatted_response = json.dumps(mcp_response, indent=2)
                response_message = f"✅ Successfully executed '{tool_request.tool_name}'\n\nResponse:\n{formatted_response}"
            
            state["messages"] = [AIMessage(content=response_message)]
            logger.info("Response formatted successfully")
            
        except Exception as e:
            state["error"] = f"Error formatting response: {str(e)}"
            logger.error(state["error"])
        
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors and provide helpful feedback"""
        error_message = state.get("error", "Unknown error occurred")
        
        response_message = f"❌ Error: {error_message}\n\nPlease check:\n• Your GitHub token has the required permissions\n• Repository exists and you have access\n• Parameters are correctly formatted\n• Tool name is spelled correctly"
        
        state["messages"] = [AIMessage(content=response_message)]
        return state
    
    def run_with_params(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Run the agent with direct tool name and parameters"""
        tool_request = GitHubToolRequest(tool_name=tool_name, parameters=parameters)
        
        initial_state = {
            "messages": [HumanMessage(content=f"Execute {tool_name} with {parameters}")],
            "tool_request": tool_request,
            "mcp_response": None,
            "error": None
        }
        
        try:
            # Skip parsing and go directly to connection
            final_state = self.graph.invoke(initial_state, {"start": "connect_mcp"})
            return final_state["messages"][-1].content
        finally:
            self.cleanup()
    
    def run(self, user_request: str) -> str:
        """Run the agent with a user request"""
        initial_state = {
            "messages": [HumanMessage(content=user_request)],
            "tool_request": None,
            "mcp_response": None,
            "error": None
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state["messages"][-1].content
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            logger.info("Cleaned up GitHub MCP server process")

def parse_json_params(json_string: str) -> Dict[str, Any]:
    """Parse JSON parameters from string"""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON parameters: {e}")
        sys.exit(1)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="GitHub MCP Agent - Execute GitHub tools via MCP")
    parser.add_argument("--tool", "-t", type=str, help="GitHub tool name to execute")
    parser.add_argument("--params", "-p", type=str, help="JSON string of parameters for the tool")
    parser.add_argument("--request", "-r", type=str, help="Natural language request")
    parser.add_argument("--list-tools", action="store_true", help="List available GitHub tools")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable is not set")
        print("Please set it with: export GROQ_API_KEY=your_groq_api_key")
        sys.exit(1)
    
    if not github_token:
        print("❌ Error: GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set")
        print("Please set it with: export GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token")
        sys.exit(1)
    
    if args.list_tools:
        print("📋 Available GitHub MCP Tools:")
        print("\n🗂️  REPOSITORIES:")
        repo_tools = ["create_branch", "list_branches", "get_file_contents", "create_or_update_file", 
                     "delete_file", "list_commits", "get_commit", "list_tags", "get_tag", 
                     "search_repositories", "search_code", "create_repository", "fork_repository", "push_files"]
        for tool in repo_tools:
            print(f"   • {tool}")
        
        print("\n🐛 ISSUES:")
        issue_tools = ["create_issue", "get_issue", "list_issues", "update_issue", "add_issue_comment", 
                      "get_issue_comments", "search_issues", "assign_copilot_to_issue"]
        for tool in issue_tools:
            print(f"   • {tool}")
        
        print("\n🔄 PULL REQUESTS:")
        pr_tools = ["create_pull_request", "get_pull_request", "list_pull_requests", "update_pull_request", 
                   "merge_pull_request", "get_pull_request_diff", "get_pull_request_files", 
                   "get_pull_request_comments", "get_pull_request_reviews", "get_pull_request_status", 
                   "search_pull_requests", "request_copilot_review"]
        for tool in pr_tools:
            print(f"   • {tool}")
        
        print("\n⚡ ACTIONS (CI/CD):")
        action_tools = ["list_workflows", "list_workflow_runs", "get_workflow_run", "cancel_workflow_run", 
                       "rerun_workflow_run", "run_workflow", "list_workflow_jobs", "get_job_logs"]
        for tool in action_tools:
            print(f"   • {tool}")
        
        print("\n🔒 SECURITY:")
        security_tools = ["list_code_scanning_alerts", "get_code_scanning_alert", "list_dependabot_alerts", 
                         "get_dependabot_alert", "list_secret_scanning_alerts", "get_secret_scanning_alert"]
        for tool in security_tools:
            print(f"   • {tool}")
        
        print("\n🔔 NOTIFICATIONS:")
        notification_tools = ["list_notifications", "get_notification_details", "dismiss_notification", 
                             "mark_all_notifications_read"]
        for tool in notification_tools:
            print(f"   • {tool}")
        
        print("\n💬 DISCUSSIONS:")
        discussion_tools = ["list_discussions", "get_discussion", "get_discussion_comments", 
                           "list_discussion_categories"]
        for tool in discussion_tools:
            print(f"   • {tool}")
        
        print("\n👤 CONTEXT & USERS:")
        user_tools = ["get_me", "search_users", "search_orgs"]
        for tool in user_tools:
            print(f"   • {tool}")
        
        return
    
    agent = GitHubMCPAgent(groq_api_key)
    
    try:
        if args.tool and args.params:
            # Direct tool execution
            parameters = parse_json_params(args.params)
            print(f"🚀 Executing GitHub tool: {args.tool}")
            print(f"📝 Parameters: {json.dumps(parameters, indent=2)}")
            print("-" * 50)
            response = agent.run_with_params(args.tool, parameters)
            print(response)
        
        elif args.request:
            # Natural language request
            print(f"🚀 Processing request: {args.request}")
            print("-" * 50)
            response = agent.run(args.request)
            print(response)
        
        else:
            # Interactive mode
            print("🎯 GitHub MCP Agent - Interactive Mode")
            print("Enter 'quit' to exit, 'help' for usage examples")
            
            while True:
                print("\nChoose an option:")
                print("1. Execute tool directly (tool_name + JSON params)")
                print("2. Natural language request")
                print("3. Quit")
                
                choice = input("\nChoice (1-3): ").strip()
                
                if choice == "1":
                    tool_name = input("Tool name: ").strip()
                    if not tool_name:
                        print("❌ Tool name cannot be empty")
                        continue
                    
                    params_str = input("Parameters (JSON): ").strip()
                    if not params_str:
                        params_str = "{}"
                    
                    try:
                        parameters = json.loads(params_str)
                        print(f"\n🚀 Executing: {tool_name}")
                        print("-" * 50)
                        response = agent.run_with_params(tool_name, parameters)
                        print(response)
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON: {e}")
                
                elif choice == "2":
                    request = input("Your request: ").strip()
                    if not request:
                        print("❌ Request cannot be empty")
                        continue
                    
                    print(f"\n🚀 Processing: {request}")
                    print("-" * 50)
                    response = agent.run(request)
                    print(response)
                
                elif choice == "3":
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()