from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import subprocess

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
    tool_selection_mode: bool