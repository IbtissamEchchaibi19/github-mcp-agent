# GitHub MCP Tools Agent

## Overview

The GitHub MCP Tools Agent is an intelligent conversational interface that bridges natural language queries with GitHub's Model Context Protocol (MCP) server capabilities. This system enables users to discover, configure, and execute GitHub operations through an intuitive chat interface powered by advanced semantic search and automated parameter collection.

## Architecture Overview

<p align="center">
  <img src="https://github.com/IbtissamEchchaibi19/github-mcp-agent/blob/6c2eab9b0ea31370058a99181c249260d803381e/project%20represnatation.png" alt="Architecture Diagram" width="900"/>
</p>

you can see the demo here in this video 
[Watch Video](https://drive.google.com/file/d/YOUR_FILE_ID/view)

## Core Components

### 1. Enhanced Tool Storage System

The storage system implements a sophisticated semantic enrichment pipeline that transforms raw GitHub MCP tool definitions into semantically rich, searchable entities.

**Strategic Database Design:**

The system employs a multi-layered approach to tool storage:

- **Functional Classification**: Tools are categorized by action type (create, read, update, delete, list, search) and resource type (issue, pr, repository, workflow, user, etc.)

- **Semantic Enrichment**: Each tool is enhanced with use cases, keywords, complexity assessment, and relationship mapping

- **Hybrid Search Architecture**: Combines vector similarity search with metadata filtering for precise tool discovery

- **Intent Extraction**: Natural language processing to understand user intent and apply appropriate filters
images
<p align="center">
  <img src="https://github.com/IbtissamEchchaibi19/github-mcp-agent/blob/6c2eab9b0ea31370058a99181c249260d803381e/weiavate1.png" alt="Architecture Diagram" width="900"/>
</p>
<p align="center">
  <img src="https://github.com/IbtissamEchchaibi19/github-mcp-agent/blob/6c2eab9b0ea31370058a99181c249260d803381e/weaivate2.png" alt="Architecture Diagram" width="900"/>
</p>

### 2. Conversational Workflow Engine

Built on LangGraph, the workflow engine manages complex multi-turn conversations with state persistence.

**Workflow Phases:**
1. **Tool Discovery Phase**: Semantic search and ranking
2. **Tool Selection Phase**: Number-based selection from results
3. **Parameter Collection Phase**: Interactive parameter gathering
4. **Execution Phase**: MCP server communication and result handling

**Intelligent Routing Logic:**
The system dynamically routes conversations based on context, handling tool discovery, selection, parameter collection, and execution phases seamlessly.

### 3. MCP Client Integration

Handles communication with the GitHub MCP server using JSON-RPC protocol over Docker containers, managing server initialization, tool execution, and response processing.

## Technology Stack

### Large Language Model
- **Model**: Llama 3.3 70B (via Groq API)
- **Advantages**:
  - High-quality reasoning for tool selection
  - Excellent natural language understanding
  - Fast inference through Groq's optimized infrastructure
  - Strong parameter validation and conversation flow management

### Vector Database
- **Database**: Weaviate
- **Embedding Model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Features**:
  - Hybrid search combining vector similarity and metadata filtering
  - Real-time intent extraction and query optimization
  - Scalable architecture supporting thousands of tools
  - Advanced aggregation for analytics and tool discovery

### GitHub MCP Server

The GitHub MCP (Model Context Protocol) server provides standardized access to GitHub's API through a unified interface.

**Advantages:**
- **Standardized Protocol**: Consistent interface across different GitHub operations
- **Authentication Management**: Automated token handling and security
- **Rate Limiting**: Built-in protection against API abuse
- **Error Handling**: Comprehensive error reporting and recovery
- **Type Safety**: Structured tool definitions with parameter validation

### Docker MCP Toolkits (Beta)

This project leverages Docker's new MCP Toolkits feature, which provides:

- **Containerized MCP Servers**: Isolated, reproducible execution environments
- **Easy Deployment**: Single command deployment with docker run
- **Environment Management**: Secure handling of API keys and configuration
- **Resource Isolation**: Protected execution without affecting host system
- **Version Management**: Consistent tool versions across environments
<p align="center">
  <img src="https://github.com/IbtissamEchchaibi19/github-mcp-agent/blob/6c2eab9b0ea31370058a99181c249260d803381e/dockerdashb.png" alt="Architecture Diagram" width="900"/>
</p>
<p align="center">
  <img src="https://github.com/IbtissamEchchaibi19/github-mcp-agent/blob/6c2eab9b0ea31370058a99181c249260d803381e/apiconfiguration.png" alt="Architecture Diagram" width="900"/>
</p>
## Database Storage Strategy

### Tool Enhancement Pipeline

The system transforms raw tool definitions through multiple enhancement layers:

1. **Raw Tool Extraction**: Parse JSON definitions from GitHub MCP documentation
2. **Semantic Classification**: Automatically categorize tools by function and resource type
3. **Metadata Enrichment**: Generate use cases, keywords, and complexity scores
4. **Relationship Discovery**: Identify related tools and workflow dependencies
5. **Vector Embedding**: Generate semantic embeddings for similarity search
6. **Hybrid Storage**: Store structured metadata alongside vector representations

### Search Optimization

The database implements a three-tier search approach:

1. **Intent Analysis**: Extract action type and resource type from natural language
2. **Vector Search**: Find semantically similar tools using embeddings
3. **Relevance Scoring**: Re-rank results based on multiple factors including vector similarity, use case matching, keyword overlap, and complexity preference

## Prerequisites

### System Requirements
- Python 3.12+
- Docker Desktop with MCP Toolkits support
- 4GB+ RAM (for embedding models)
- Internet connection for API access

### Required API Keys
- **GROQ_API_KEY**: For Llama 3.3 70B access
- **GITHUB_PERSONAL_ACCESS_TOKEN**: For GitHub operations
- **WEAVIATE_API_KEY**: (Optional, for Weaviate Cloud)

### Dependencies
- LangGraph for workflow orchestration
- Weaviate for vector database
- Sentence Transformers for embeddings
- ChatGroq for LLM integration 

## Installation and Setup

### Step 1: Environment Setup

1. Clone the repository and navigate to the project directory

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. install Github MCP server using Docker Desktop with MCP Toolkits
images
5. run the Github MCP server using :
```bash
docker mcp gateway run
``` 
### Step 2: Environment Configuration

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_key_here  
EMBEDDING_MODEL=all-MiniLM-L6-v2
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=api key
LANGSMITH_PROJECT="autogithubagent"

```

### Step 3: Data Preparation

1. Extract GitHub MCP tools from the official documentation
2. Parse the tools section into `github_mcp_tools.json`
3. Ensure the JSON file is in the project root directory

### Step 4: Vector Database Setup

**Option A: Local Weaviate (Recommended for development)**
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  cr.weaviate.io/semitechnologies/weaviate:1.24.0
```

**Option B: Weaviate Cloud**
- Sign up at https://console.weaviate.cloud
- Create a cluster and obtain API key
- Update WEAVIATE_URL and WEAVIATE_API_KEY in .env

### Step 5: Initialize the System

Run the initialization script to populate the vector database:
```bash
python Weaviate.py
```

This will:
- Load and enhance tool definitions
- Generate semantic embeddings
- Create Weaviate schema
- Store enriched tools in the database

## Execution Steps

### Interactive Chat Mode

1. Start the interactive session:
```bash
python chatbot.py
```

2. Follow the conversational flow:
   - **Search**: Enter natural language queries like "create an issue" or "list repositories"
   - **Select**: Choose a tool by entering its number (1-5)
   - **Configure**: Provide parameters when prompted
   - **Execute**: Type "execute" to run the tool

### Method 3: LangGraph Studio Integration

The project includes LangGraph Studio compatibility for visual workflow debugging and monitoring.

## License

This project is designed for educational and development purposes. Ensure compliance with GitHub API terms of service and Docker usage policies when deploying in production environments.
