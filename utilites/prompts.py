from typing import Annotated, TypedDict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from prompts import clasifyUserneedPrompt, ClarifyPrompt
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

categories_dict = {

}

class ClassificationState(TypedDict):
    text: str
    category: str
    classification_explanation: str  # Explanation of why this category was chosen
    clarification_question: str
    human_feedback: str  # New field for human response
    iteration_count: int
    needs_clarification: bool  # Flag to track if we're waiting for human input
    needs_confirmation: bool  # Flag to track if we need user confirmation
    user_confirmed: bool  # Flag to track if user confirmed the classification


def classify(state: ClassificationState) -> ClassificationState:
    logger.info(f"Starting classification for text: '{state['text'][:50]}...'")
    
    # If we have human feedback, incorporate it into the classification
    user_text = state["text"]
    if state.get("human_feedback"):
        user_text += f"\n\nAdditional context from user: {state['human_feedback']}"
        logger.info("Including human feedback in classification")
    
    # Use your existing prompt with proper formatting
    prompt_with_text = clasifyUserneedPrompt.format(categories_dict=categories_dict)
    full_prompt = f"{prompt_with_text}\n\nUser prompt: \"{user_text}\""
    
    try:
        logger.info("Calling LLM for classification...")
        response = llm.invoke(full_prompt)
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM Response: {response_content[:200]}...")
        
        # Parse the response to extract category and explanation
        # Look for explicit classification patterns first
        response_lower = response_content.lower()
        
        # Check if the response explicitly says it's unclear/vague/unknown
        unclear_indicators = [
            "unclear", "vague", "too general", "cannot determine", 
            "need more information", "ambiguous", "unknown", "cannot classify"
        ]
        
        is_unclear = any(indicator in response_lower for indicator in unclear_indicators)
        
        if is_unclear:
            state["category"] = "Unclear"
            state["needs_clarification"] = True
            state["needs_confirmation"] = False
            logger.info("Classification determined as unclear based on response content")
        else:
            # Look for actual classification in structured format
            found_category = None
            
            # Try to find "Classification: [Category]" pattern first
            if "classification:" in response_lower:
                classification_line = ""
                for line in response_content.split('\n'):
                    if "classification:" in line.lower():
                        classification_line = line
                        break
                
                if classification_line:
                    for category in categories_dict.keys():
                        if category.lower() in classification_line.lower():
                            found_category = category
                            break
            
            # Fallback: look for category names in the full response
            if not found_category:
                category_mentions = []
                for category in categories_dict.keys():
                    if category.lower() in response_lower:
                        category_mentions.append(category)
                
                # If only one category is mentioned, use it
                if len(category_mentions) == 1:
                    found_category = category_mentions[0]
                elif len(category_mentions) > 1:
                    # Multiple categories mentioned - likely unclear
                    found_category = None
            
            if found_category:
                state["category"] = found_category
                state["classification_explanation"] = response_content
                state["needs_clarification"] = False
                state["needs_confirmation"] = True
                logger.info(f"Classification successful: {found_category}")
            else:
                state["category"] = "Unclear"
                state["needs_clarification"] = True
                state["needs_confirmation"] = False
                logger.warning("No clear category found in response")
            
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        state["category"] = "Error"
        state["needs_clarification"] = True
        state["needs_confirmation"] = False
    
    return state


def clarify(state: ClassificationState) -> ClassificationState:
    logger.info("Starting clarification process...")
    
    # Increment iteration counter
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    try:
        # Use your existing ClarifyPrompt from prompts module
        logger.info("Calling LLM for clarification using ClarifyPrompt...")
        response = llm.invoke(ClarifyPrompt)
        
        clarification_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Clarification response: {clarification_content[:100]}...")
        
        state["clarification_question"] = clarification_content
        state["needs_clarification"] = True
        
        # Don't interrupt here - let the flow continue to wait_for_human
        
    except Exception as e:
        logger.error(f"Error during clarification: {e}")
        state["clarification_question"] = "Could you please provide more details about your request?"
        state["needs_clarification"] = True
    
    return state


def wait_for_human(state: ClassificationState) -> ClassificationState:
    """This node interrupts execution to wait for human input"""
    logger.info("Waiting for human input...")
    
    # Determine what we're asking for
    if state.get("needs_confirmation"):
        # Asking for confirmation of classification
        interrupt_data = {
            "type": "confirmation",
            "category": state.get("category", "Unknown"),
            "explanation": state.get("classification_explanation", ""),
            "question": "Is this classification correct? (Yes/No)"
        }
    else:
        # Asking for clarification
        interrupt_data = {
            "type": "clarification",
            "question": state.get("clarification_question", "Please provide more details"),
            "current_category": state.get("category", "Unknown"),
            "iteration": state.get("iteration_count", 0)
        }
    
    # This will cause the graph to pause and wait for human input
    human_response = interrupt(value=interrupt_data)
    
    # Process the human's response
    if human_response:
        human_response_str = str(human_response).lower().strip()
        
        if state.get("needs_confirmation"):
            # Handle confirmation response
            if human_response_str in ["yes", "y", "correct", "true"]:
                state["user_confirmed"] = True
                state["needs_confirmation"] = False
                logger.info("User confirmed the classification")
            else:
                state["user_confirmed"] = False
                state["needs_confirmation"] = False
                state["needs_clarification"] = True
                logger.info("User rejected classification, needs clarification")
        else:
            # Handle clarification response - store feedback and reset flags
            state["human_feedback"] = str(human_response)
            state["needs_clarification"] = False  # Clear this flag
            state["needs_confirmation"] = False   # Clear this flag too
            logger.info(f"Received human feedback: {human_response}")
    
    return state


def should_continue(state: ClassificationState) -> str:
    """Determine the next step in the workflow"""
    category = state.get("category", "")
    iteration_count = state.get("iteration_count", 0)
    needs_clarification = state.get("needs_clarification", False)
    needs_confirmation = state.get("needs_confirmation", False)
    user_confirmed = state.get("user_confirmed", False)
    human_feedback = state.get("human_feedback", "")
    
    logger.info(f"Decision point - Category: {category}, Iterations: {iteration_count}")
    logger.info(f"Needs clarification: {needs_clarification}, Needs confirmation: {needs_confirmation}")
    logger.info(f"User confirmed: {user_confirmed}, Has feedback: {bool(human_feedback)}")
    
    # Prevent infinite loops
    if iteration_count >= 3:
        logger.warning("Max iterations reached, forcing end")
        return "end"
    
    # If user confirmed the classification, we're done
    if user_confirmed:
        logger.info("Route: end (user confirmed)")
        return "end"
    
    # If we need confirmation, wait for human input
    if needs_confirmation:
        logger.info("Route: wait_for_human (needs confirmation)")
        return "wait_for_human"
    
    # If we need clarification, go to clarify node first
    if needs_clarification:
        logger.info("Route: clarify (needs clarification)")
        return "clarify"
    
    # If we just received human feedback and need to re-classify
    if human_feedback and not needs_confirmation and not needs_clarification:
        logger.info("Route: classify (processing human feedback)")
        return "classify"
    
    # If classification was unclear, ask for clarification
    if category in ["Unclear", "", "Error"]:
        logger.info("Route: clarify (unclear category)")
        return "clarify"
    
    logger.info("Route: end (default)")
    return "end"


# Build the agent with human-in-the-loop
logger.info("Building classification agent with human-in-the-loop...")
agent = StateGraph(ClassificationState)

# Add nodes
agent.add_node("classify", classify)
agent.add_node("clarify", clarify)
agent.add_node("wait_for_human", wait_for_human)

# Add edges
agent.add_edge(START, "classify")
agent.add_conditional_edges(
    "classify", 
    should_continue, 
    {"clarify": "clarify", "wait_for_human": "wait_for_human", "end": END}
)
agent.add_edge("clarify", "wait_for_human")  # Always go to wait_for_human after clarify
agent.add_conditional_edges(
    "wait_for_human", 
    should_continue,
    {"classify": "classify", "clarify": "clarify", "end": END}
)  # Route based on what we need to do after human input

# Compile with interrupts enabled and memory checkpointer
memory = MemorySaver()
classifyAgent = agent.compile(checkpointer=memory, interrupt_after=["wait_for_human"])
logger.info("Classification agent built successfully with human-in-the-loop")


def interactive_test_with_hitl():
    """Interactive testing with human-in-the-loop support"""
    logger.info("=== INTERACTIVE CLASSIFICATION AGENT WITH HUMAN-IN-THE-LOOP ===")
    print("\n🤖 Classification Agent Ready!")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            # Get user input
            user_input = input("📝 Enter your request: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                print("⚠️ Please enter some text")
                continue
            
            # Create initial state
            config = {"configurable": {"thread_id": "test-thread"}}
            initial_state = {
                "text": user_input,
                "category": "",
                "classification_explanation": "",
                "clarification_question": "",
                "human_feedback": "",
                "iteration_count": 0,
                "needs_clarification": False,
                "needs_confirmation": False,
                "user_confirmed": False
            }
            
            logger.info(f"Processing user input: {user_input}")
            
            # Start the workflow
            result = classifyAgent.invoke(initial_state, config=config)
            
            # Handle interruptions (requests for clarification or confirmation)
            while classifyAgent.get_state(config).next:
                current_state = classifyAgent.get_state(config)
                state_values = current_state.values if hasattr(current_state, 'values') else {}
                
                logger.info(f"Current state in interactive loop: {state_values.get('category', 'No category')}")
                logger.info(f"Needs confirmation: {state_values.get('needs_confirmation', False)}")
                logger.info(f"Needs clarification: {state_values.get('needs_clarification', False)}")
                
                # Determine what type of input we need
                if state_values.get("needs_confirmation"):
                    # Show classification and ask for confirmation
                    print(f"\n🎯 CLASSIFICATION RESULT:")
                    print(f"📋 Category: {state_values.get('category', 'Unknown')}")
                    if state_values.get('classification_explanation'):
                        explanation_lines = state_values['classification_explanation'].split('\n')[:3]  # First 3 lines
                        for line in explanation_lines:
                            if line.strip():
                                print(f"💭 {line.strip()}")
                    print(f"\n❓ Is this classification correct? (Yes/No)")
                    
                    human_response = input("✅ Your answer: ").strip()
                    
                elif state_values.get("needs_clarification"):
                    # Show clarification question
                    print(f"\n❓ CLARIFICATION NEEDED:")
                    clarification = state_values.get('clarification_question', '')
                    if clarification:
                        print(f"🤔 {clarification}")
                    else:
                        print("🤔 Could you please provide more details about your request? For example:")
                        print("   - Is it related to writing or debugging code?")
                        print("   - Is it about configuring infrastructure or cloud services?")
                        print("   - Are you looking for documentation or guides?")
                        print("   - Do you need help with databases, APIs, or system logs?")
                    
                    human_response = input("\n💭 Your response: ").strip()
                else:
                    # Fallback
                    print(f"\n❓ NEED MORE INFO:")
                    print("🤔 Please provide more details about your request")
                    
                    human_response = input("\n💭 Your response: ").strip()
                
                if human_response.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    return
                
                # Resume with human feedback
                result = classifyAgent.invoke(
                    Command(resume=human_response), 
                    config=config
                )
            
            # Display final results
            print(f"\n🎯 FINAL RESULT:")
            print(f"📋 Category: {result['category']}")
            if result.get('human_feedback'):
                print(f"💡 Used feedback: {result['human_feedback']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during interactive test: {e}")
            print(f"❌ Error: {e}")


def simple_test():
    """Simple test without interaction"""
    logger.info("Running simple test...")
    
    config = {"configurable": {"thread_id": "simple-test"}}
    test_state = {
        "text": "Help me with something",  # Intentionally vague
        "category": "",
        "clarification_question": "",
        "human_feedback": "",
        "iteration_count": 0,
        "needs_clarification": False
    }
    
    result = classifyAgent.invoke(test_state, config=config)
    
    print("\n=== SIMPLE TEST RESULTS ===")
    print(f"Input: {test_state['text']}")
    print(f"Category: {result['category']}")
    if result.get('clarification_question'):
        print(f"Clarification needed: {result['clarification_question']}")
    print("========================")


# Example usage
if __name__ == "__main__":
    logger.info("Starting classification agent with human-in-the-loop...")
    
    print("\n🚀 Classification Agent with Human-in-the-Loop")
    print("1. Interactive mode (with clarification)")
    print("2. Simple test")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        interactive_test_with_hitl()
    else:
        simple_test()