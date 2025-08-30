import os
import json
import subprocess
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MCPClient:
    """Handles all MCP server communication"""
    
    def __init__(self):
        self.request_id = 1
    
    def run_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
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