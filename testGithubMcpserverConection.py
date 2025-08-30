
import json
import subprocess
import logging
import time
import random
import string
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class CreateBranchTester:
    def __init__(self):
        self.process = None
        self.request_id = 1
        # Repository details
        self.owner = "IbtissamEchchaibi19"
        self.repo = "MCP-powered-IT-Engineering-Copilot"
    
    def connect_to_server(self) -> bool:
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
            
            time.sleep(3)
            
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                logger.error(f"Server failed to start: {stderr_output}")
                return False
            
            logger.info("Connected to MCP server successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def send_request(self, method: str, params=None):
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
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                return None
            
            return json.loads(response_line.strip())
            
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None
    
    def initialize_server(self) -> bool:
        """Initialize the MCP server"""
        logger.info("Initializing server...")
        
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "create-branch-tester",
                "version": "1.0.0"
            }
        }
        
        response = self.send_request("initialize", params)
        
        if response and "result" in response:
            logger.info("Server initialized successfully")
            return True
        else:
            logger.error(f"Failed to initialize server: {response}")
            return False
    
    def generate_branch_name(self) -> str:
        """Generate a unique branch name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        return f"test-branch-{timestamp}-{random_suffix}"
    
    def create_branch(self, branch_name: str = None, from_branch: str = None) -> bool:
        """Test creating a branch"""
        if not branch_name:
            branch_name = self.generate_branch_name()
        
        logger.info(f"Creating branch '{branch_name}' in {self.owner}/{self.repo}")
        if from_branch:
            logger.info(f"Source branch: {from_branch}")
        
        arguments = {
            "owner": self.owner,
            "repo": self.repo,
            "branch": branch_name
        }
        
        if from_branch:
            arguments["from_branch"] = from_branch
        
        params = {
            "name": "create_branch",
            "arguments": arguments
        }
        
        response = self.send_request("tools/call", params)
        
        if response and "result" in response:
            logger.info("Branch created successfully!")
            
            # Log response details if available
            if "content" in response["result"]:
                for content_item in response["result"]["content"]:
                    if content_item.get("type") == "text":
                        logger.info(f"Response: {content_item.get('text', '')}")
            
            return True
            
        elif response and "error" in response:
            error = response["error"]
            logger.error(f"Failed to create branch: {error.get('message', 'Unknown error')}")
            logger.debug(f"Error code: {error.get('code', 'No code')}")
            return False
        else:
            logger.error(f"Unexpected response: {response}")
            return False
    
    def run_test(self, branch_name: str = None, from_branch: str = None) -> bool:
        """Run the create branch test"""
        logger.info("Starting create branch test")
        logger.info(f"Target repository: {self.owner}/{self.repo}")
        
        if not self.connect_to_server():
            return False
        
        if not self.initialize_server():
            return False
        
        success = self.create_branch(branch_name, from_branch)
        
        if success:
            logger.info("Create branch test completed successfully!")
        else:
            logger.error("Create branch test failed")
        
        return success
    
    def cleanup(self):
        """Clean up resources"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            logger.info("Cleaned up server process")

def main():
    """Main function"""
    tester = CreateBranchTester()
    
    try:
        # Test with auto-generated branch name
        success = tester.run_test()
        
        # Optional: Test with custom branch name and source branch
        # success = tester.run_test("my-feature-branch", "main")
        
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit(main())