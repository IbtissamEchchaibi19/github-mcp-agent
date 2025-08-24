from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.types import Command, interrupt
import os
import logging
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
llmapi = os.getenv("GROQ_API_KEY")
logger.info(f"API Key loaded: {'Yes' if llmapi else 'No'}")

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    api_key=llmapi
)
logger.info("LLM initialized successfully")

# GitHub MCP Server Tool Categories and Mappings
GITHUB_TOOLS_MAPPING = {
    "repository_management": {
        "description": "Repository operations, file management, and code browsing",
        "tools": [
            "create_repository",
            "fork_repository", 
            "get_file_contents",
            "create_or_update_file",
            "delete_file",
            "push_files",
            "create_branch",
            "list_branches",
            "list_commits",
            "get_commit",
            "list_tags",
            "get_tag",
            "search_code",
            "search_repositories"
        ],
        "keywords": ["repository", "repo", "file", "code", "branch", "commit", "upload", "download", "create repo", "fork", "clone", "search code"]
    },
    
    "issues_management": {
        "description": "Issue tracking, bug reports, and task management",
        "tools": [
            "create_issue",
            "get_issue",
            "update_issue",
            "list_issues",
            "search_issues",
            "add_issue_comment",
            "get_issue_comments",
            "assign_copilot_to_issue"
        ],
        "keywords": ["issue", "bug", "task", "ticket", "problem", "report", "comment", "assign", "track"]
    },
    
    "pull_requests": {
        "description": "Pull request creation, review, and management",
        "tools": [
            "create_pull_request",
            "get_pull_request",
            "update_pull_request",
            "list_pull_requests",
            "search_pull_requests",
            "merge_pull_request",
            "get_pull_request_diff",
            "get_pull_request_files",
            "get_pull_request_comments",
            "get_pull_request_reviews",
            "get_pull_request_status",
            "update_pull_request_branch",
            "create_pending_pull_request_review",
            "add_comment_to_pending_review",
            "submit_pending_pull_request_review",
            "delete_pending_pull_request_review",
            "create_and_submit_pull_request_review",
            "request_copilot_review"
        ],
        "keywords": ["pull request", "pr", "merge", "review", "code review", "diff", "changes", "merge request"]
    },
    
    "ci_cd_workflows": {
        "description": "GitHub Actions, CI/CD pipelines, and workflow automation",
        "tools": [
            "list_workflows",
            "run_workflow",
            "get_workflow_run",
            "list_workflow_runs",
            "list_workflow_jobs",
            "get_job_logs",
            "get_workflow_run_logs",
            "get_workflow_run_usage",
            "list_workflow_run_artifacts",
            "download_workflow_run_artifact",
            "cancel_workflow_run",
            "rerun_workflow_run",
            "rerun_failed_jobs",
            "delete_workflow_run_logs"
        ],
        "keywords": ["workflow", "actions", "ci/cd", "build", "deploy", "pipeline", "automation", "job", "artifact", "logs"]
    },
    
    "security_monitoring": {
        "description": "Security alerts, code scanning, and vulnerability management",
        "tools": [
            "list_code_scanning_alerts",
            "get_code_scanning_alert",
            "list_dependabot_alerts",
            "get_dependabot_alert",
            "list_secret_scanning_alerts",
            "get_secret_scanning_alert"
        ],
        "keywords": ["security", "vulnerability", "alert", "scanning", "dependabot", "secret", "code scanning", "cve"]
    },
    
    "discussions_community": {
        "description": "Community discussions and Q&A management",
        "tools": [
            "list_discussions",
            "get_discussion",
            "get_discussion_comments",
            "list_discussion_categories"
        ],
        "keywords": ["discussion", "community", "q&a", "forum", "conversation", "comment"]
    },
    
    "notifications_management": {
        "description": "Notification management and subscription handling",
        "tools": [
            "list_notifications",
            "get_notification_details",
            "dismiss_notification",
            "manage_notification_subscription",
            "manage_repository_notification_subscription",
            "mark_all_notifications_read"
        ],
        "keywords": ["notification", "alert", "subscription", "inbox", "unread", "dismiss"]
    },
    
    "user_org_search": {
        "description": "User profiles, organization search, and context information",
        "tools": [
            "get_me",
            "search_users",
            "search_orgs"
        ],
        "keywords": ["user", "profile", "organization", "org", "search", "member", "team", "account"]
    },
    
    "ai_automation": {
        "description": "AI-powered automation and Copilot integration",
        "tools": [
            "create_pull_request_with_copilot"
        ],
        "keywords": ["copilot", "ai", "automated", "generate", "intelligent", "assistant", "auto-generate"]
    }
}

# Enhanced classification prompts
GITHUB_TOOL_CLASSIFICATION_PROMPT = """
You are a GitHub MCP Server tool classification expert. Your task is to analyze user prompts and determine which GitHub MCP tools would be most appropriate to fulfill their request.

Available GitHub MCP Tool Categories:
{categories_info}

Guidelines for Classification:
1. Analyze the user's intent and identify the primary GitHub operation they want to perform
2. Consider the specific GitHub entities mentioned (repositories, issues, PRs, workflows, etc.)
3. Look for action words that indicate specific operations (create, update, search, list, merge, etc.)
4. If the request involves multiple categories, prioritize the primary action
5. If the request is too vague or unclear, classify as "Unclear"

Classification Format:
- Category: [One of the available categories]
- Tools: [List of specific tools that would be used]
- Confidence: [High/Medium/Low]
- Explanation: [Brief explanation of why these tools were selected]

User prompt: "{user_prompt}"

Provide your classification:
"""

GITHUB_TOOL_CLARIFICATION_PROMPT = """
The user's request needs clarification to determine the appropriate GitHub MCP tools. 

Please ask a specific question to help clarify:
1. What specific GitHub operation they want to perform
2. Which repository or repositories are involved
3. What type of GitHub entity they're working with (issues, PRs, files, etc.)
4. Whether they need read-only access or want to make changes

Keep your question concise and focused on getting actionable information.
"""

class GitHubMCPClassificationState(TypedDict):
    text: str
    category: str
    classification_explanation: str
    recommended_tools: List[str]
    tool_parameters: Dict[str, Any]
    clarification_question: str
    human_feedback: str
    iteration_count: int
    needs_clarification: bool
    needs_confirmation: bool
    user_confirmed: bool
    confidence_level: str

def get_categories_info():
    """Generate formatted category information for the prompt"""
    info = []
    for category, details in GITHUB_TOOLS_MAPPING.items():
        tools_list = ", ".join(details["tools"][:5])  # Show first 5 tools
        if len(details["tools"]) > 5:
            tools_list += f" (and {len(details['tools']) - 5} more)"
        
        info.append(f"- {category}: {details['description']}")
        info.append(f"  Example tools: {tools_list}")
        info.append(f"  Keywords: {', '.join(details['keywords'][:8])}")  # Show first 8 keywords
        info.append("")
    
    return "\n".join(info)

def classify_github_tools(state: GitHubMCPClassificationState) -> GitHubMCPClassificationState:
    """Classify user prompt and recommend GitHub MCP tools"""
    logger.info(f"Starting GitHub MCP tool classification for text: '{state['text'][:50]}...'")
    
    # Include human feedback if available
    user_text = state["text"]
    if state.get("human_feedback"):
        user_text += f"\n\nAdditional context from user: {state['human_feedback']}"
        logger.info("Including human feedback in classification")
    
    # Prepare the classification prompt
    categories_info = get_categories_info()
    full_prompt = GITHUB_TOOL_CLASSIFICATION_PROMPT.format(
        categories_info=categories_info,
        user_prompt=user_text
    )
    
    try:
        logger.info("Calling LLM for GitHub tool classification...")
        response = llm.invoke(full_prompt)
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM Response: {response_content[:200]}...")
        
        # Parse the response to extract classification information
        response_lower = response_content.lower()
        
        # Check for unclear indicators
        unclear_indicators = [
            "unclear", "vague", "too general", "cannot determine", 
            "need more information", "ambiguous", "unknown", "cannot classify"
        ]
        
        is_unclear = any(indicator in response_lower for indicator in unclear_indicators)
        
        if is_unclear:
            state["category"] = "Unclear"
            state["needs_clarification"] = True
            state["needs_confirmation"] = False
            state["confidence_level"] = "Low"
            logger.info("Classification determined as unclear based on response content")
        else:
            # Try to extract category from response
            found_category = None
            found_tools = []
            confidence = "Medium"
            
            # Look for category mentions
            for category in GITHUB_TOOLS_MAPPING.keys():
                if category.replace("_", " ") in response_lower or category in response_lower:
                    found_category = category
                    break
            
            # If no direct category match, try keyword matching
            if not found_category:
                category_scores = {}
                for category, details in GITHUB_TOOLS_MAPPING.items():
                    score = 0
                    for keyword in details["keywords"]:
                        if keyword.lower() in user_text.lower():
                            score += 1
                    if score > 0:
                        category_scores[category] = score
                
                if category_scores:
                    found_category = max(category_scores, key=category_scores.get)
                    confidence = "High" if category_scores[found_category] > 2 else "Medium"
            
            # Extract tools from response
            if "tools:" in response_lower:
                tools_section = response_content.split("Tools:")[1].split("\n")[0] if "Tools:" in response_content else ""
                # Simple extraction - could be improved with more sophisticated parsing
                for category, details in GITHUB_TOOLS_MAPPING.items():
                    for tool in details["tools"]:
                        if tool in tools_section:
                            found_tools.append(tool)
            
            # If we found a category, get its tools
            if found_category:
                if not found_tools:  # If no specific tools extracted, use category tools
                    found_tools = GITHUB_TOOLS_MAPPING[found_category]["tools"][:3]  # Top 3 most relevant
                
                state["category"] = found_category
                state["recommended_tools"] = found_tools
                state["classification_explanation"] = response_content
                state["confidence_level"] = confidence
                state["needs_clarification"] = False
                state["needs_confirmation"] = True
                logger.info(f"Classification successful: {found_category} with {len(found_tools)} tools")
            else:
                state["category"] = "Unclear"
                state["needs_clarification"] = True
                state["needs_confirmation"] = False
                state["confidence_level"] = "Low"
                logger.warning("No clear category found in response")
    
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        state["category"] = "Error"
        state["recommended_tools"] = []
        state["needs_clarification"] = True
        state["needs_confirmation"] = False
        state["confidence_level"] = "Low"
    
    return state

def clarify_github_tools(state: GitHubMCPClassificationState) -> GitHubMCPClassificationState:
    """Generate clarification question for GitHub tool selection"""
    logger.info("Starting GitHub tool clarification process...")
    
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    try:
        logger.info("Calling LLM for GitHub tool clarification...")
        response = llm.invoke(GITHUB_TOOL_CLARIFICATION_PROMPT)
        
        clarification_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Clarification response: {clarification_content[:100]}...")
        
        state["clarification_question"] = clarification_content
        state["needs_clarification"] = True
        
    except Exception as e:
        logger.error(f"Error during clarification: {e}")
        state["clarification_question"] = "Could you please provide more details about what GitHub operation you'd like to perform? For example: creating/updating files, managing issues or pull requests, running workflows, or checking security alerts?"
        state["needs_clarification"] = True
    
    return state

def wait_for_human(state: GitHubMCPClassificationState) -> GitHubMCPClassificationState:
    """Handle human interaction for GitHub tool selection"""
    logger.info("Waiting for human input...")
    
    if state.get("needs_confirmation"):
        interrupt_data = {
            "type": "confirmation",
            "category": state.get("category", "Unknown"),
            "tools": state.get("recommended_tools", []),
            "confidence": state.get("confidence_level", "Medium"),
            "explanation": state.get("classification_explanation", ""),
            "question": f"I've identified this as a '{state.get('category', 'Unknown').replace('_', ' ').title()}' task. The recommended GitHub MCP tools are: {', '.join(state.get('recommended_tools', [])[:5])}. Is this correct? (Yes/No)"
        }
    else:
        interrupt_data = {
            "type": "clarification",
            "question": state.get("clarification_question", "Please provide more details about your GitHub operation"),
            "current_category": state.get("category", "Unknown"),
            "iteration": state.get("iteration_count", 0)
        }
    
    human_response = interrupt(value=interrupt_data)
    
    if human_response:
        human_response_str = str(human_response).lower().strip()
        
        if state.get("needs_confirmation"):
            if human_response_str in ["yes", "y", "correct", "true"]:
                state["user_confirmed"] = True
                state["needs_confirmation"] = False
                logger.info("User confirmed the GitHub tool classification")
            else:
                state["user_confirmed"] = False
                state["needs_confirmation"] = False
                state["needs_clarification"] = True
                logger.info("User rejected classification, needs clarification")
        else:
            state["human_feedback"] = str(human_response)
            state["needs_clarification"] = False
            state["needs_confirmation"] = False
            logger.info(f"Received human feedback: {human_response}")
    
    return state

def should_continue(state: GitHubMCPClassificationState) -> str:
    """Determine the next step in the GitHub tool classification workflow"""
    category = state.get("category", "")
    iteration_count = state.get("iteration_count", 0)
    needs_clarification = state.get("needs_clarification", False)
    needs_confirmation = state.get("needs_confirmation", False)
    user_confirmed = state.get("user_confirmed", False)
    human_feedback = state.get("human_feedback", "")
    
    logger.info(f"Decision point - Category: {category}, Iterations: {iteration_count}")
    logger.info(f"Needs clarification: {needs_clarification}, Needs confirmation: {needs_confirmation}")
    logger.info(f"User confirmed: {user_confirmed}, Has feedback: {bool(human_feedback)}")
    
    if iteration_count >= 3:
        logger.warning("Max iterations reached, forcing end")
        return "end"
    
    if user_confirmed:
        logger.info("Route: end (user confirmed)")
        return "end"
    
    if needs_confirmation:
        logger.info("Route: wait_for_human (needs confirmation)")
        return "wait_for_human"
    
    if needs_clarification:
        logger.info("Route: clarify (needs clarification)")
        return "clarify"
    
    if human_feedback and not needs_confirmation and not needs_clarification:
        logger.info("Route: classify (processing human feedback)")
        return "classify"
    
    if category in ["Unclear", "", "Error"]:
        logger.info("Route: clarify (unclear category)")
        return "clarify"
    
    logger.info("Route: end (default)")
    return "end"

# Build the GitHub MCP tool classification agent
logger.info("Building GitHub MCP tool classification agent...")
agent = StateGraph(GitHubMCPClassificationState)

# Add nodes
agent.add_node("classify", classify_github_tools)
agent.add_node("clarify", clarify_github_tools)
agent.add_node("wait_for_human", wait_for_human)

# Add edges
agent.add_edge(START, "classify")
agent.add_conditional_edges(
    "classify", 
    should_continue, 
    {"clarify": "clarify", "wait_for_human": "wait_for_human", "end": END}
)
agent.add_edge("clarify", "wait_for_human")
agent.add_conditional_edges(
    "wait_for_human", 
    should_continue,
    {"classify": "classify", "clarify": "clarify", "end": END}
)

# Compile with memory and interrupts
memory = MemorySaver()
github_mcp_agent = agent.compile(checkpointer=memory, interrupt_after=["wait_for_human"])
logger.info("GitHub MCP tool classification agent built successfully")

def interactive_github_mcp_test():
    """Interactive testing for GitHub MCP tool recommendations"""
    logger.info("=== GITHUB MCP TOOL CLASSIFICATION AGENT ===")
    print("\n🔧 GitHub MCP Tool Classification Agent Ready!")
    print("I'll help you identify the right GitHub MCP server tools for your task.")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            user_input = input("📝 Describe what you want to do with GitHub: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                print("⚠️ Please describe what you want to do")
                continue
            
            config = {"configurable": {"thread_id": "github-mcp-test"}}
            initial_state = {
                "text": user_input,
                "category": "",
                "classification_explanation": "",
                "recommended_tools": [],
                "tool_parameters": {},
                "clarification_question": "",
                "human_feedback": "",
                "iteration_count": 0,
                "needs_clarification": False,
                "needs_confirmation": False,
                "user_confirmed": False,
                "confidence_level": "Medium"
            }
            
            logger.info(f"Processing GitHub MCP request: {user_input}")
            
            result = github_mcp_agent.invoke(initial_state, config=config)
            
            # Handle interruptions
            while github_mcp_agent.get_state(config).next:
                current_state = github_mcp_agent.get_state(config)
                state_values = current_state.values if hasattr(current_state, 'values') else {}
                
                if state_values.get("needs_confirmation"):
                    print(f"\n🎯 GITHUB MCP TOOL RECOMMENDATION:")
                    print(f"📋 Category: {state_values.get('category', 'Unknown').replace('_', ' ').title()}")
                    print(f"🔧 Recommended Tools:")
                    for i, tool in enumerate(state_values.get('recommended_tools', [])[:5], 1):
                        print(f"   {i}. {tool}")
                    print(f"🎯 Confidence: {state_values.get('confidence_level', 'Medium')}")
                    
                    if state_values.get('classification_explanation'):
                        explanation_lines = state_values['classification_explanation'].split('\n')[:2]
                        for line in explanation_lines:
                            if line.strip() and 'explanation:' in line.lower():
                                print(f"💭 {line.strip()}")
                    
                    print(f"\n❓ Are these the right tools for your GitHub task? (Yes/No)")
                    human_response = input("✅ Your answer: ").strip()
                    
                elif state_values.get("needs_clarification"):
                    print(f"\n❓ NEED MORE INFORMATION:")
                    clarification = state_values.get('clarification_question', '')
                    if clarification:
                        print(f"🤔 {clarification}")
                    else:
                        print("🤔 Could you please be more specific about your GitHub operation?")
                        print("   Examples:")
                        print("   - 'Create a new issue in my repository'")
                        print("   - 'Merge a pull request'") 
                        print("   - 'Check workflow logs for failed builds'")
                        print("   - 'Update a file in the main branch'")
                    
                    human_response = input("\n💭 Your response: ").strip()
                else:
                    print(f"\n❓ CLARIFICATION NEEDED:")
                    print("🤔 Please provide more details about your GitHub request")
                    human_response = input("\n💭 Your response: ").strip()
                
                if human_response.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    return
                
                result = github_mcp_agent.invoke(
                    Command(resume=human_response), 
                    config=config
                )
            
            # Display final results
            print(f"\n🎯 FINAL GITHUB MCP TOOL RECOMMENDATION:")
            print(f"📋 Category: {result['category'].replace('_', ' ').title()}")
            print(f"🔧 Tools to use: {', '.join(result.get('recommended_tools', [])[:5])}")
            if result.get('confidence_level'):
                print(f"🎯 Confidence Level: {result['confidence_level']}")
            
            # Show tool usage example
            if result.get('recommended_tools'):
                print(f"\n📖 Example Usage:")
                print(f"   Use the GitHub MCP server with tools: {result['recommended_tools'][0]}")
                if result['category'] in GITHUB_TOOLS_MAPPING:
                    print(f"   Description: {GITHUB_TOOLS_MAPPING[result['category']]['description']}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during interactive test: {e}")
            print(f"❌ Error: {e}")

def simple_github_test():
    """Simple test for GitHub MCP tool classification"""
    logger.info("Running simple GitHub MCP test...")
    
    config = {"configurable": {"thread_id": "simple-github-test"}}
    test_state = {
        "text": "I want to create a pull request",
        "category": "",
        "recommended_tools": [],
        "clarification_question": "",
        "human_feedback": "",
        "iteration_count": 0,
        "needs_clarification": False,
        "confidence_level": "Medium"
    }
    
    result = github_mcp_agent.invoke(test_state, config=config)
    
    print("\n=== SIMPLE GITHUB MCP TEST RESULTS ===")
    print(f"Input: {test_state['text']}")
    print(f"Category: {result['category']}")
    print(f"Recommended Tools: {result.get('recommended_tools', [])}")
    print(f"Confidence: {result.get('confidence_level', 'Medium')}")
    if result.get('clarification_question'):
        print(f"Clarification needed: {result['clarification_question']}")
    print("=====================================")

# Example usage
if __name__ == "__main__":
    logger.info("Starting GitHub MCP tool classification agent...")
    
    print("\n🚀 GitHub MCP Tool Classification Agent")
    print("1. Interactive mode (with tool recommendations)")
    print("2. Simple test")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        interactive_github_mcp_test()
    else:
        simple_github_test()