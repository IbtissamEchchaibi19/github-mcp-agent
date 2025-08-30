import os
from chatbot import EnhancedGitHubMCPChatbot

def main():
    """Main function"""
    JSON_FILE_PATH = "github_mcp_tools.json"
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    
    try:
        chatbot = EnhancedGitHubMCPChatbot(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY
        )
        
        chatbot.run_interactive_session()
        
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main()