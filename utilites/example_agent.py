# example_agent.py
"""
Example AI agent using the MCP client package.
Demonstrates how to use the reusable MCP client for GitHub operations.
"""

import logging
from mcp_client import github_mcp_client, mcp_client, MCPClientError
from datetime import datetime
import random
import string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubAgent:
    """
    AI Agent that performs GitHub operations using MCP.
    """
    
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
    
    def generate_branch_name(self, prefix: str = "feature") -> str:
        """Generate a unique branch name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        return f"{prefix}-{timestamp}-{random_suffix}"
    
    def create_feature_branch(self, feature_name: str = None, from_branch: str = None):
        """Create a feature branch"""
        if not feature_name:
            feature_name = self.generate_branch_name("feature")
        
        try:
            with github_mcp_client(self.owner, self.repo) as client:
                logger.info(f"Creating feature branch: {feature_name}")
                result = client.create_branch(feature_name, from_branch)
                logger.info("Feature branch created successfully!")
                return result
        except MCPClientError as e:
            logger.error(f"Failed to create branch: {e}")
            raise
    
    def create_bug_report(self, title: str, description: str, labels: list = None):
        """Create a bug report issue"""
        if not labels:
            labels = ["bug"]
        
        try:
            with github_mcp_client(self.owner, self.repo) as client:
                logger.info(f"Creating bug report: {title}")
                result = client.create_issue(title, description, labels)
                logger.info("Bug report created successfully!")
                return result
        except MCPClientError as e:
            logger.error(f"Failed to create bug report: {e}")
            raise
    
    def list_open_issues(self):
        """List all open issues"""
        try:
            with github_mcp_client(self.owner, self.repo) as client:
                logger.info("Fetching open issues...")
                result = client.list_issues("open")
                logger.info(f"Found open issues")
                return result
        except MCPClientError as e:
            logger.error(f"Failed to list issues: {e}")
            raise
    
    def comment_on_issue(self, issue_number: int, comment: str):
        """Add a comment to an issue"""
        try:
            with github_mcp_client(self.owner, self.repo) as client:
                logger.info(f"Adding comment to issue #{issue_number}")
                result = client.add_comment(issue_number, comment)
                logger.info("Comment added successfully!")
                return result
        except MCPClientError as e:
            logger.error(f"Failed to add comment: {e}")
            raise
    
    def get_available_tools(self):
        """Get list of all available GitHub tools"""
        try:
            with mcp_client() as client:
                tools = client.list_tools()
                logger.info(f"Available tools: {len(tools)}")
                for tool in tools[:10]:  # Show first 10
                    logger.info(f"  - {tool.get('name')}: {tool.get('description', '')[:50]}...")
                return tools
        except MCPClientError as e:
            logger.error(f"Failed to get tools: {e}")
            raise

def main():
    """Example usage of the GitHub Agent"""
    
    # Initialize the agent with your repository
    agent = GitHubAgent("IbtissamEchchaibi19", "MCP-powered-IT-Engineering-Copilot")
    
    try:
        # Example 1: Create a feature branch
        logger.info("=== Creating Feature Branch ===")
        branch_result = agent.create_feature_branch("ai-improvements")
        
        # Example 2: Create a bug report
        logger.info("=== Creating Bug Report ===")
        bug_result = agent.create_bug_report(
            title="AI agent connection timeout",
            description="The AI agent sometimes experiences timeout when connecting to MCP server.\n\nSteps to reproduce:\n1. Start agent\n2. Wait for connection\n3. Observe timeout after 30 seconds",
            labels=["bug", "ai-agent", "priority:medium"]
        )
        
        # Example 3: List open issues
        logger.info("=== Listing Open Issues ===")
        issues_result = agent.list_open_issues()
        
        # Example 4: Get available tools
        logger.info("=== Available Tools ===")
        tools = agent.get_available_tools()
        
        logger.info("All operations completed successfully!")
        
    except MCPClientError as e:
        logger.error(f"Agent operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()