import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage

def has_recent_tool_list(messages: List[BaseMessage]) -> bool:
    """Check if the last bot message contains a tool list"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content.lower()
            if ("choose a number" in content and 
                any(str(i) + "." in content for i in range(1, 6))):
                return True
            break
    return False

def extract_task_from_query(query: str) -> str:
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

def format_tool_details_with_collection_start(tool_details: Dict[str, Any]) -> str:
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

def show_help():
    """Show help information"""
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

def check_environment():
    """Check required environment variables"""
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print("⚠️  Warning: GITHUB_PERSONAL_ACCESS_TOKEN not set!")
        print("Set it with: export GITHUB_PERSONAL_ACCESS_TOKEN=your_token")
        print()