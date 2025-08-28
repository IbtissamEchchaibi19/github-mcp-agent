import json
import subprocess
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass

class MCPClient:
    """
    A reusable client for interacting with MCP servers via Docker.
    Handles connection, initialization, and tool execution.
    """
    
    def __init__(self, server_timeout: int = 5):
        self.process = None
        self.request_id = 1
        self.server_timeout = server_timeout
        self.initialized = False
        
    def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            logger.info("Connecting to MCP server...")
            
            self.process = subprocess.Popen(
                ['docker', 'mcp', 'gateway', 'run'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            time.sleep(self.server_timeout)
            
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                raise MCPClientError(f"Server failed to start: {stderr_output}")
            
            logger.info("Connected to MCP server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.cleanup()
            raise MCPClientError(f"Connection failed: {e}")
    
    def disconnect(self):
        """Disconnect from the MCP server"""
        self.cleanup()
        logger.info("Disconnected from MCP server")
    
    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server"""
        if not self.process:
            raise MCPClientError("Not connected to server")
        
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
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            response_line = self.process.stdout.readline()
            if not response_line:
                raise MCPClientError(f"No response received for method: {method}")
            response = json.loads(response_line.strip())
            if "error" in response:
                error = response["error"]
                raise MCPClientError(f"Server error: {error.get('message', 'Unknown error')}")
            return response    
        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise MCPClientError(f"Request failed: {e}")
    
    def initialize(self) -> bool:
        """Initialize the MCP server"""
        if self.initialized:
            return True
        logger.info("Initializing MCP server...")
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "mcp-ai-agent",
                "version": "1.0.0"
            }
        }
        try:
            response = self.send_request("initialize", params)
            if "result" in response:
                self.initialized = True
                logger.info("MCP server initialized successfully")
                return True
            else:
                raise MCPClientError("Initialize failed: no result in response")
                
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        try:
            response = self.send_request("tools/list")
            return response.get("result", {}).get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """Get list of available resources"""
        try:
            response = self.send_request("resources/list")
            return response.get("result", {}).get("resources", [])
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            raise
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with arguments"""
        logger.info(f"Calling tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        
        try:
            response = self.send_request("tools/call", params)
            result = response.get("result", {})
            
            logger.info(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tools = self.list_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
        self.initialized = False

# Context manager for automatic cleanup
@contextmanager
def mcp_client(server_timeout: int = 5):
    """Context manager for MCP client with automatic cleanup"""
    client = MCPClient(server_timeout=server_timeout)
    try:
        client.connect()
        client.initialize()
        yield client
    finally:
        client.disconnect()

# Convenience class for GitHub operations
class GitHubMCPClient(MCPClient):
    """
    Specialized MCP client for GitHub operations.
    Provides convenient methods for common GitHub tasks.
    """
    
    def __init__(self, owner: str, repo: str, server_timeout: int = 5):
        super().__init__(server_timeout)
        self.owner = owner
        self.repo = repo
    
    def create_branch(self, branch_name: str, from_branch: str = None) -> Dict[str, Any]:
        """Create a new branch"""
        arguments = {
            "owner": self.owner,
            "repo": self.repo,
            "branch": branch_name
        }
        
        if from_branch:
            arguments["from_branch"] = from_branch
        
        return self.call_tool("create_branch", arguments)
    
    def create_issue(self, title: str, body: str = None, labels: List[str] = None) -> Dict[str, Any]:
        """Create a new issue"""
        arguments = {
            "owner": self.owner,
            "repo": self.repo,
            "title": title
        }
        
        if body:
            arguments["body"] = body
        if labels:
            arguments["labels"] = labels
        
        return self.call_tool("create_issue", arguments)
    
    def add_comment(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """Add comment to an issue"""
        arguments = {
            "owner": self.owner,
            "repo": self.repo,
            "issue_number": issue_number,
            "comment": comment
        }
        
        return self.call_tool("add_issue_comment", arguments)
    
    def list_issues(self, state: str = "open") -> Dict[str, Any]:
        """List repository issues"""
        arguments = {
            "owner": self.owner,
            "repo": self.repo,
            "state": state
        }
        
        return self.call_tool("list_issues", arguments)

# Context manager for GitHub operations
@contextmanager
def github_mcp_client(owner: str, repo: str, server_timeout: int = 5):
    """Context manager for GitHub MCP client with automatic cleanup"""
    client = GitHubMCPClient(owner, repo, server_timeout)
    try:
        client.connect()
        client.initialize()
        yield client
    finally:
        client.disconnect()