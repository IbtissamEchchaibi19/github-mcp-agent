import json
import os
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wc
from weaviate.classes.query import Filter, MetadataQuery
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class EnhancedGitHubTool:
    """Enhanced schema for GitHub MCP tools with semantic enrichment"""
    tool_name: str
    section: str
    full_qualified_name: str
    
    # Functional metadata
    description: str
    action_type: str  # "create", "read", "update", "delete", "list", "search"
    resource_type: str  # "issue", "pr", "repository", "workflow", "user", etc.
    
    # Parameters and requirements
    required_params: List[str]
    optional_params: List[str]
    parameters_schema: Dict[str, Any]
    param_count: int
    
    # Semantic enrichment
    use_cases: List[str]
    keywords: List[str]
    complexity_level: str  # "simple", "intermediate", "advanced"
    
    # Relationships
    related_tools: List[str]
    prerequisite_tools: List[str]
    common_workflows: List[str]
    
    # Enhanced content for retrieval
    enriched_content: str
    searchable_text: str
    
    # Metadata for filtering
    requires_auth: bool = True
    api_rate_limited: bool = True
    is_mutation: bool = False  # True for create/update/delete operations
    
    # System metadata
    chunk_id: str = ""
    created_at: str = ""

class GitHubMCPEnhancer:
    """Enhances raw GitHub MCP tool data with semantic information"""
    
    def __init__(self):
        self.action_keywords = {
            'create': ['create', 'new', 'add', 'open', 'make', 'generate', 'post', 'submit'],
            'read': ['get', 'show', 'fetch', 'retrieve', 'view', 'display', 'find', 'download'],
            'update': ['update', 'edit', 'modify', 'change', 'patch', 'set', 'rerun'],
            'delete': ['delete', 'remove', 'cancel', 'dismiss', 'close'],
            'list': ['list', 'get_all', 'fetch_all', 'show_all', 'search'],
            'search': ['search', 'find', 'query', 'filter']
        }
        
        self.resource_keywords = {
            'issue': ['issue', 'ticket', 'bug', 'feature', 'task'],
            'pr': ['pull', 'request', 'merge', 'review', 'branch'],
            'repository': ['repo', 'repository', 'project', 'code'],
            'workflow': ['workflow', 'action', 'ci', 'cd', 'pipeline', 'job'],
            'user': ['user', 'member', 'team', 'organization', 'org'],
            'notification': ['notification', 'alert', 'message'],
            'gist': ['gist', 'snippet', 'code_snippet'],
            'discussion': ['discussion', 'forum', 'talk'],
            'security': ['security', 'vulnerability', 'advisory', 'dependabot', 'secret']
        }
        
        self.common_use_cases = {
            'create_issue': ['report bug', 'request feature', 'track task', 'create ticket'],
            'list_issues': ['view issues', 'check status', 'review tickets', 'project management'],
            'create_pull_request': ['propose changes', 'submit code', 'code review', 'merge request'],
            'merge_pull_request': ['deploy changes', 'accept review', 'complete feature'],
            'list_workflows': ['check ci status', 'monitor builds', 'view deployments'],
            'run_workflow': ['trigger deployment', 'run tests', 'execute pipeline'],
            'search_repositories': ['find projects', 'discover code', 'research'],
            'create_repository': ['start project', 'initialize repo', 'new codebase']
        }
    
    def classify_action_type(self, tool_name: str, description: str) -> str:
        """Classify the action type based on tool name and description"""
        text = f"{tool_name} {description}".lower()
        
        for action_type, keywords in self.action_keywords.items():
            if any(keyword in text for keyword in keywords):
                return action_type
        
        # Default classification based on common patterns
        if tool_name.startswith(('create', 'add', 'open', 'make')):
            return 'create'
        elif tool_name.startswith(('get', 'show', 'fetch', 'retrieve')):
            return 'read'
        elif tool_name.startswith(('update', 'edit', 'modify', 'set')):
            return 'update'
        elif tool_name.startswith(('delete', 'remove', 'cancel')):
            return 'delete'
        elif tool_name.startswith(('list', 'search', 'find')):
            return 'list'
        
        return 'read'  # Default
    
    def extract_resource_type(self, tool_name: str, section: str, description: str) -> str:
        """Extract the primary resource type"""
        text = f"{tool_name} {section} {description}".lower()
        
        for resource_type, keywords in self.resource_keywords.items():
            if any(keyword in text for keyword in keywords):
                return resource_type
        
        # Fallback to section-based classification
        section_lower = section.lower()
        if 'issue' in section_lower:
            return 'issue'
        elif 'pull' in section_lower:
            return 'pr'
        elif 'repo' in section_lower:
            return 'repository'
        elif 'action' in section_lower or 'workflow' in section_lower:
            return 'workflow'
        elif 'user' in section_lower or 'org' in section_lower:
            return 'user'
        
        return 'repository'  # Default
    
    def generate_use_cases(self, tool_name: str, action_type: str, resource_type: str) -> List[str]:
        """Generate common use cases for the tool"""
        key = f"{action_type}_{resource_type}"
        base_cases = self.common_use_cases.get(key, [])
        
        # Add generic use cases
        generic_cases = [
            f"{action_type} {resource_type}",
            f"{action_type} github {resource_type}",
            f"github {resource_type} {action_type}"
        ]
        
        # Tool-specific cases
        tool_specific = []
        if 'workflow' in tool_name:
            tool_specific.extend(['ci/cd', 'automation', 'deployment'])
        if 'review' in tool_name:
            tool_specific.extend(['code review', 'peer review', 'quality assurance'])
        if 'comment' in tool_name:
            tool_specific.extend(['feedback', 'discussion', 'collaboration'])
        
        return list(set(base_cases + generic_cases + tool_specific))[:10]  # Limit to 10
    
    def extract_keywords(self, tool_name: str, description: str, section: str) -> List[str]:
        """Extract relevant keywords for search"""
        text = f"{tool_name} {description} {section}".lower()
        
        # Basic keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Add synonyms and variations
        keywords = set(words)
        
        # Add common GitHub terminology
        github_terms = ['github', 'git', 'repo', 'repository', 'branch', 'commit', 'merge']
        keywords.update(term for term in github_terms if term in text)
        
        # Add action-specific terms
        if any(term in text for term in ['create', 'new', 'add']):
            keywords.update(['create', 'new', 'add', 'make', 'generate'])
        if any(term in text for term in ['get', 'fetch', 'retrieve']):
            keywords.update(['get', 'fetch', 'retrieve', 'show', 'view'])
        
        return list(keywords)[:20]  # Limit to 20 keywords
    
    def assess_complexity(self, parameters: Dict[str, Any]) -> str:
        """Assess tool complexity based on parameters"""
        if not parameters:
            return 'simple'
        
        required_count = sum(1 for p in parameters.values() 
                           if isinstance(p, str) and 'required' in p.lower())
        total_params = len(parameters)
        
        if total_params <= 2 and required_count <= 1:
            return 'simple'
        elif total_params <= 5 and required_count <= 3:
            return 'intermediate'
        else:
            return 'advanced'
    
    def find_related_tools(self, tool_name: str, section: str, all_tools: List[str]) -> List[str]:
        """Find related tools based on naming patterns and functionality"""
        related = []
        tool_base = tool_name.replace('_', ' ').split()[0]
        
        for other_tool in all_tools:
            if other_tool == tool_name:
                continue
            
            # Same section tools
            if other_tool.startswith(tool_base) or tool_base in other_tool:
                related.append(other_tool)
            
            # Complementary actions
            if 'create' in tool_name and 'list' in other_tool:
                related.append(other_tool)
            elif 'get' in tool_name and 'update' in other_tool:
                related.append(other_tool)
        
        return related[:5]  # Limit to 5 related tools
    
    def enhance_tool(self, section: str, tool_name: str, tool_data: Dict[str, Any], 
                    all_tools: List[str]) -> EnhancedGitHubTool:
        """Create an enhanced tool object with semantic enrichment"""
        
        description = tool_data.get("description", "")
        parameters = tool_data.get("parameters", {})
        
        # Extract parameter information
        required_params = []
        optional_params = []
        for param_name, param_info in parameters.items():
            if isinstance(param_info, str):
                if 'required' in param_info.lower():
                    required_params.append(param_name)
                else:
                    optional_params.append(param_name)
        
        # Classify tool
        action_type = self.classify_action_type(tool_name, description)
        resource_type = self.extract_resource_type(tool_name, section, description)
        
        # Generate semantic enrichment
        use_cases = self.generate_use_cases(tool_name, action_type, resource_type)
        keywords = self.extract_keywords(tool_name, description, section)
        complexity_level = self.assess_complexity(parameters)
        
        # Find relationships
        related_tools = self.find_related_tools(tool_name, section, all_tools)
        
        # Create enriched content
        enriched_content = self.create_enriched_content(
            tool_name, description, use_cases, keywords, parameters, section
        )
        
        # Create searchable text
        searchable_text = f"{section} {tool_name} {description} {' '.join(use_cases)} {' '.join(keywords)}"
        
        # Generate unique chunk ID
        chunk_id = hashlib.md5(f"{section}_{tool_name}".encode()).hexdigest()[:12]
        
        return EnhancedGitHubTool(
            tool_name=tool_name,
            section=section,
            full_qualified_name=f"{section}.{tool_name}",
            description=description,
            action_type=action_type,
            resource_type=resource_type,
            required_params=required_params,
            optional_params=optional_params,
            parameters_schema=parameters,
            param_count=len(parameters),
            use_cases=use_cases,
            keywords=keywords,
            complexity_level=complexity_level,
            related_tools=related_tools,
            prerequisite_tools=[],  # Can be enhanced later
            common_workflows=[],    # Can be enhanced later
            enriched_content=enriched_content,
            searchable_text=searchable_text,
            is_mutation=action_type in ['create', 'update', 'delete'],
            chunk_id=chunk_id,
            created_at=datetime.now().isoformat()
        )
    
    def create_enriched_content(self, tool_name: str, description: str, 
                              use_cases: List[str], keywords: List[str], 
                              parameters: Dict[str, Any], section: str) -> str:
        """Create semantically rich content for better retrieval"""
        
        content_parts = [
            f"Section: {section}",
            f"Tool: {tool_name}",
            f"Description: {description}",
            f"Common use cases: {', '.join(use_cases[:5])}",
            f"Keywords: {', '.join(keywords[:10])}",
        ]
        
        if parameters:
            content_parts.append("Parameters:")
            for param_name, param_info in parameters.items():
                if isinstance(param_info, str):
                    content_parts.append(f"  - {param_name}: {param_info}")
                else:
                    content_parts.append(f"  - {param_name}: {str(param_info)[:100]}")
        
        return "\n".join(content_parts)

class EnhancedEmbeddingService:
    """Enhanced text embeddings with preprocessing"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            console.print(f"[yellow]Loading embedding model: {self.model_name}[/yellow]")
            self.model = SentenceTransformer(self.model_name)
            console.print(f"[green]✓ Embedding model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load embedding model: {e}[/red]")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better embeddings"""
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text with preprocessing"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        processed_text = self.preprocess_text(text)
        return self.model.encode(processed_text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.model.encode(processed_texts).tolist()

class EnhancedWeaviateService:
    """Enhanced Weaviate service with advanced querying capabilities"""
    
    def __init__(self, url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.collection_name = "EnhancedGitHubMCPToolsStorage"
        self._connect()
    
    def _connect(self):
        """Connect to Weaviate instance"""
        try:
            console.print(f"[yellow]Connecting to Weaviate at {self.url}[/yellow]")
            
            if ".weaviate.cloud" in self.url:
                if not self.api_key:
                    raise ValueError("API key is required for Weaviate Cloud connections")
                
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                )
            else:
                if self.url.startswith("http://"):
                    host_port = self.url.replace("http://", "")
                elif self.url.startswith("https://"):
                    host_port = self.url.replace("https://", "")
                else:
                    host_port = self.url
                
                if ":" in host_port:
                    host, port = host_port.split(":")
                    port = int(port)
                else:
                    host = host_port
                    port = 8080
                
                self.client = weaviate.connect_to_local(host=host, port=port)
            
            if self.client.is_ready():
                console.print("[green]✓ Connected to Weaviate successfully[/green]")
            else:
                raise ConnectionError("Weaviate is not ready")
                
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to Weaviate: {e}[/red]")
            raise
    
    def __del__(self):
        """Ensure client is properly closed"""
        if self.client:
            self.client.close()
    
    def create_enhanced_schema(self):
        """Create the enhanced schema for GitHub MCP tools"""
        try:
            if self.client.collections.exists(self.collection_name):
                console.print(f"[yellow]Deleting existing collection: {self.collection_name}[/yellow]")
                self.client.collections.delete(self.collection_name)
            
            console.print(f"[yellow]Creating enhanced collection: {self.collection_name}[/yellow]")
            
            collection = self.client.collections.create(
                name=self.collection_name,
                description="Enhanced GitHub MCP Tools with semantic enrichment",
                properties=[
                    # Core identifiers
                    wc.Property(name="toolName", data_type=wc.DataType.TEXT),
                    wc.Property(name="section", data_type=wc.DataType.TEXT),
                    wc.Property(name="fullQualifiedName", data_type=wc.DataType.TEXT),
                    
                    # Functional metadata
                    wc.Property(name="description", data_type=wc.DataType.TEXT),
                    wc.Property(name="actionType", data_type=wc.DataType.TEXT),
                    wc.Property(name="resourceType", data_type=wc.DataType.TEXT),
                    
                    # Parameters
                    wc.Property(name="requiredParams", data_type=wc.DataType.TEXT_ARRAY),
                    wc.Property(name="optionalParams", data_type=wc.DataType.TEXT_ARRAY),
                    wc.Property(name="parametersSchema", data_type=wc.DataType.TEXT),
                    wc.Property(name="paramCount", data_type=wc.DataType.INT),
                    
                    # Semantic enrichment
                    wc.Property(name="useCases", data_type=wc.DataType.TEXT_ARRAY),
                    wc.Property(name="keywords", data_type=wc.DataType.TEXT_ARRAY),
                    wc.Property(name="complexityLevel", data_type=wc.DataType.TEXT),
                    
                    # Relationships
                    wc.Property(name="relatedTools", data_type=wc.DataType.TEXT_ARRAY),
                    
                    # Content
                    wc.Property(name="enrichedContent", data_type=wc.DataType.TEXT),
                    wc.Property(name="searchableText", data_type=wc.DataType.TEXT),
                    
                    # Metadata
                    wc.Property(name="requiresAuth", data_type=wc.DataType.BOOL),
                    wc.Property(name="isMutation", data_type=wc.DataType.BOOL),
                    wc.Property(name="chunkId", data_type=wc.DataType.TEXT),
                    wc.Property(name="createdAt", data_type=wc.DataType.TEXT),
                ],
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.COSINE
                )
            )
            
            console.print("[green]✓ Enhanced schema created successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to create schema: {e}[/red]")
            raise
    
    def store_enhanced_tools(self, tools: List[EnhancedGitHubTool], embeddings: List[List[float]]):
        """Store enhanced tools with their embeddings"""
        try:
            console.print(f"[yellow]Storing {len(tools)} enhanced tools in Weaviate[/yellow]")
            
            collection = self.client.collections.get(self.collection_name)
            
            inserted_count = 0
            for i, (tool, embedding) in enumerate(zip(tools, embeddings)):
                try:
                    properties = {
                        "toolName": tool.tool_name,
                        "section": tool.section,
                        "fullQualifiedName": tool.full_qualified_name,
                        "description": tool.description,
                        "actionType": tool.action_type,
                        "resourceType": tool.resource_type,
                        "requiredParams": tool.required_params,
                        "optionalParams": tool.optional_params,
                        "parametersSchema": json.dumps(tool.parameters_schema),
                        "paramCount": tool.param_count,
                        "useCases": tool.use_cases,
                        "keywords": tool.keywords,
                        "complexityLevel": tool.complexity_level,
                        "relatedTools": tool.related_tools,
                        "enrichedContent": tool.enriched_content,
                        "searchableText": tool.searchable_text,
                        "requiresAuth": tool.requires_auth,
                        "isMutation": tool.is_mutation,
                        "chunkId": tool.chunk_id,
                        "createdAt": tool.created_at
                    }
                    
                    uuid = collection.data.insert(
                        properties=properties,
                        vector=embedding
                    )
                    
                    inserted_count += 1
                    
                    if (i + 1) % 10 == 0:
                        console.print(f"[blue]Processed {i + 1}/{len(tools)} tools[/blue]")
                        
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to insert tool {i}: {e}[/yellow]")
                    continue
            
            console.print(f"[green]✓ Successfully stored {inserted_count}/{len(tools)} enhanced tools[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to store tools: {e}[/red]")
            raise
    
    def hybrid_search(self, query_embedding: List[float], query_text: str, 
                     filters: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search with vector similarity and metadata filtering"""
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Build where filter using the correct v4 API
            where_filter = None
            if filters:
                conditions = []
                
                if 'action_type' in filters:
                    conditions.append(Filter.by_property("actionType").equal(filters['action_type']))
                
                if 'resource_type' in filters:
                    conditions.append(Filter.by_property("resourceType").equal(filters['resource_type']))
                
                if 'complexity_level' in filters:
                    conditions.append(Filter.by_property("complexityLevel").equal(filters['complexity_level']))
                
                if 'is_mutation' in filters:
                    conditions.append(Filter.by_property("isMutation").equal(filters['is_mutation']))
                
                if 'section' in filters:
                    conditions.append(Filter.by_property("section").equal(filters['section']))
                
                # Combine conditions with AND using Filter.all_of or & operator
                if len(conditions) == 1:
                    where_filter = conditions[0]
                elif len(conditions) > 1:
                    where_filter = Filter.all_of(conditions)
            
            # Define return properties
            return_properties = [
                "toolName", "section", "fullQualifiedName", "description",
                "actionType", "resourceType", "requiredParams", "optionalParams",
                "parametersSchema", "useCases", "keywords", "complexityLevel",
                "relatedTools", "enrichedContent", "isMutation", "chunkId"
            ]
            
            # Perform vector search using correct v4 API
            query_params = {
                "near_vector": query_embedding,
                "limit": limit * 2,  # Get more results for re-ranking
                "return_metadata": MetadataQuery(certainty=True),
                "return_properties": return_properties
            }
            
            # Add filters if they exist
            if where_filter:
                query_params["filters"] = where_filter
            
            response = collection.query.near_vector(**query_params)
            
            # Convert to dict format and add compatibility scoring
            results = []
            for obj in response.objects:
                result = {
                    "toolName": obj.properties.get("toolName"),
                    "section": obj.properties.get("section"),
                    "fullQualifiedName": obj.properties.get("fullQualifiedName"),
                    "description": obj.properties.get("description"),
                    "actionType": obj.properties.get("actionType"),
                    "resourceType": obj.properties.get("resourceType"),
                    "requiredParams": obj.properties.get("requiredParams", []),
                    "optionalParams": obj.properties.get("optionalParams", []),
                    "parametersSchema": obj.properties.get("parametersSchema"),
                    "useCases": obj.properties.get("useCases", []),
                    "keywords": obj.properties.get("keywords", []),
                    "complexityLevel": obj.properties.get("complexityLevel"),
                    "relatedTools": obj.properties.get("relatedTools", []),
                    "enrichedContent": obj.properties.get("enrichedContent"),
                    "isMutation": obj.properties.get("isMutation"),
                    "chunkId": obj.properties.get("chunkId"),
                    "_additional": {
                        "certainty": obj.metadata.certainty if obj.metadata else None
                    }
                }
                
                # Add relevance scoring
                result["relevance_score"] = self._calculate_relevance_score(result, query_text)
                results.append(result)
            
            # Re-rank by relevance score and return top results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            console.print(f"[red]✗ Hybrid search failed: {e}[/red]")
            raise
    
    def _calculate_relevance_score(self, tool_result: Dict[str, Any], query_text: str) -> float:
        """Calculate relevance score for re-ranking"""
        score = 0.0
        query_lower = query_text.lower()
        
        # Base similarity score (from Weaviate)
        certainty = tool_result.get("_additional", {}).get("certainty", 0)
        score += certainty * 0.4
        
        # Use case matching
        use_cases = tool_result.get("useCases", [])
        use_case_matches = sum(1 for case in use_cases if any(word in query_lower for word in case.lower().split()))
        score += (use_case_matches / max(len(use_cases), 1)) * 0.3
        
        # Keyword matching
        keywords = tool_result.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
        score += (keyword_matches / max(len(keywords), 1)) * 0.2
        
        # Complexity preference (favor simpler tools for ambiguous queries)
        complexity = tool_result.get("complexityLevel", "simple")
        if complexity == "simple":
            score += 0.1
        elif complexity == "intermediate":
            score += 0.05
        
        return score
    
    def extract_query_intent(self, query_text: str) -> Dict[str, Any]:
        """Extract intent from query text for filtering"""
        query_lower = query_text.lower()
        intent = {}
        
        # Extract action type
        if any(word in query_lower for word in ['create', 'make', 'new', 'add', 'open']):
            intent['action_type'] = 'create'
        elif any(word in query_lower for word in ['get', 'show', 'fetch', 'retrieve', 'view']):
            intent['action_type'] = 'read'
        elif any(word in query_lower for word in ['update', 'edit', 'modify', 'change']):
            intent['action_type'] = 'update'
        elif any(word in query_lower for word in ['delete', 'remove', 'cancel']):
            intent['action_type'] = 'delete'
        elif any(word in query_lower for word in ['list', 'search', 'find', 'all']):
            intent['action_type'] = 'list'
        
        # Extract resource type
        if any(word in query_lower for word in ['issue', 'ticket', 'bug']):
            intent['resource_type'] = 'issue'
        elif any(word in query_lower for word in ['pull request', 'pr', 'merge']):
            intent['resource_type'] = 'pr'
        elif any(word in query_lower for word in ['repository', 'repo', 'project']):
            intent['resource_type'] = 'repository'
        elif any(word in query_lower for word in ['workflow', 'action', 'ci', 'cd']):
            intent['resource_type'] = 'workflow'
        elif any(word in query_lower for word in ['user', 'member', 'team']):
            intent['resource_type'] = 'user'
        elif any(word in query_lower for word in ['notification', 'alert']):
            intent['resource_type'] = 'notification'
        elif any(word in query_lower for word in ['gist', 'snippet']):
            intent['resource_type'] = 'gist'
        elif any(word in query_lower for word in ['security', 'vulnerability']):
            intent['resource_type'] = 'security'
        
        return intent
    
    def intelligent_search(self, query_text: str, query_embedding: List[float], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform intelligent search with intent extraction and filtering"""
        try:
            # Extract intent from query
            intent = self.extract_query_intent(query_text)
            
            # Perform hybrid search with intent-based filtering
            results = self.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                filters=intent,
                limit=top_k
            )
            
            # If no results with filters, try without filters
            if not results and intent:
                console.print("[yellow]No results with intent filters, trying broader search[/yellow]")
                results = self.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    filters=None,
                    limit=top_k
                )
            
            return results
            
        except Exception as e:
            console.print(f"[red]✗ Intelligent search failed: {e}[/red]")
            raise

class EnhancedDataProcessor:
    """Enhanced data processor with semantic enrichment"""
    
    def __init__(self):
        self.enhancer = GitHubMCPEnhancer()
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]✗ Failed to load JSON: {e}[/red]")
            raise
    
    def extract_enhanced_tools(self, data: Dict[str, Any]) -> List[EnhancedGitHubTool]:
        """Extract and enhance tool chunks from JSON data"""
        tools = []
        all_tool_names = []
        
        # First pass: collect all tool names for relationship mapping
        for section_name, section_content in data.items():
            if not isinstance(section_content, dict) or not section_content:
                continue
            
            for tool_name, tool_info in section_content.items():
                if isinstance(tool_info, dict):
                    all_tool_names.append(f"{section_name}.{tool_name}")
        
        # Second pass: create enhanced tools
        for section_name, section_content in data.items():
            if not isinstance(section_content, dict) or not section_content:
                continue
            
            for tool_name, tool_info in section_content.items():
                if not isinstance(tool_info, dict):
                    continue
                
                try:
                    enhanced_tool = self.enhancer.enhance_tool(
                        section=section_name,
                        tool_name=tool_name,
                        tool_data=tool_info,
                        all_tools=all_tool_names
                    )
                    tools.append(enhanced_tool)
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to enhance tool {tool_name}: {e}[/yellow]")
                    continue
        
        console.print(f"[green]✓ Enhanced {len(tools)} tools with semantic enrichment[/green]")
        return tools

class EnhancedGitHubMCPStorage:
    """Main class that orchestrates enhanced storage and retrieval"""
    
    def __init__(self, 
                 json_file_path: str,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.json_file_path = json_file_path
        self.embedding_service = EnhancedEmbeddingService(embedding_model)
        self.weaviate_service = EnhancedWeaviateService(weaviate_url, weaviate_api_key)
        self.data_processor = EnhancedDataProcessor()
        
    def initialize_storage(self):
        """Initialize the enhanced storage system"""
        console.print(Panel.fit("🚀 Initializing Enhanced GitHub MCP Storage", style="bold blue"))
        
        # Load and process data
        console.print("\n[bold]Step 1: Loading JSON data[/bold]")
        data = self.data_processor.load_json(self.json_file_path)
        
        console.print("\n[bold]Step 2: Extracting and enhancing tools[/bold]")
        enhanced_tools = self.data_processor.extract_enhanced_tools(data)
        
        console.print("\n[bold]Step 3: Generating embeddings[/bold]")
        embedding_texts = [tool.enriched_content for tool in enhanced_tools]
        embeddings = self.embedding_service.embed_batch(embedding_texts)
        
        console.print("\n[bold]Step 4: Setting up enhanced Weaviate schema[/bold]")
        self.weaviate_service.create_enhanced_schema()
        
        console.print("\n[bold]Step 5: Storing enhanced tools in Weaviate[/bold]")
        self.weaviate_service.store_enhanced_tools(enhanced_tools, embeddings)
        
        console.print(Panel.fit("✅ Enhanced Storage Initialization Complete!", style="bold green"))
        
        return len(enhanced_tools)
    
    def search_tools(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for tools using enhanced retrieval"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Perform intelligent search
            results = self.weaviate_service.intelligent_search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            console.print(f"[red]✗ Search failed: {e}[/red]")
            raise
    
    def get_tool_by_name(self, tool_name: str, section: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a specific tool by name and optionally section"""
        try:
            collection = self.weaviate_service.client.collections.get(
                self.weaviate_service.collection_name
            )
            
            # Build filter conditions using v4 API
            conditions = [Filter.by_property("toolName").equal(tool_name)]
            if section:
                conditions.append(Filter.by_property("section").equal(section))
            
            where_filter = Filter.all_of(conditions) if len(conditions) > 1 else conditions[0]
            
            response = collection.query.fetch_objects(
                limit=1,
                filters=where_filter  # Use filters parameter instead of where
            )
            
            if response.objects:
                obj = response.objects[0]
                return {
                    "toolName": obj.properties.get("toolName"),
                    "section": obj.properties.get("section"),
                    "fullQualifiedName": obj.properties.get("fullQualifiedName"),
                    "description": obj.properties.get("description"),
                    "actionType": obj.properties.get("actionType"),
                    "resourceType": obj.properties.get("resourceType"),
                    "requiredParams": obj.properties.get("requiredParams", []),
                    "optionalParams": obj.properties.get("optionalParams", []),
                    "parametersSchema": obj.properties.get("parametersSchema"),
                    "useCases": obj.properties.get("useCases", []),
                    "keywords": obj.properties.get("keywords", []),
                    "complexityLevel": obj.properties.get("complexityLevel"),
                    "relatedTools": obj.properties.get("relatedTools", []),
                    "enrichedContent": obj.properties.get("enrichedContent"),
                    "isMutation": obj.properties.get("isMutation"),
                    "chunkId": obj.properties.get("chunkId")
                }
            
            return None
            
        except Exception as e:
            console.print(f"[red]✗ Failed to get tool by name: {e}[/red]")
            raise
    
    def get_tools_by_section(self, section: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all tools from a specific section"""
        try:
            collection = self.weaviate_service.client.collections.get(
                self.weaviate_service.collection_name
            )
            
            response = collection.query.fetch_objects(
                limit=limit,
                filters=Filter.by_property("section").equal(section)  # Use filters parameter
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "toolName": obj.properties.get("toolName"),
                    "section": obj.properties.get("section"),
                    "description": obj.properties.get("description"),
                    "actionType": obj.properties.get("actionType"),
                    "resourceType": obj.properties.get("resourceType"),
                    "complexityLevel": obj.properties.get("complexityLevel"),
                    "parametersSchema": obj.properties.get("parametersSchema")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            console.print(f"[red]✗ Failed to get tools by section: {e}[/red]")
            raise
    
    def get_related_tools(self, tool_name: str, section: Optional[str] = None, 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Get tools related to a specific tool"""
        try:
            # First get the tool to find its related tools
            tool = self.get_tool_by_name(tool_name, section)
            if not tool:
                return []
            
            related_tool_names = tool.get("relatedTools", [])
            if not related_tool_names:
                return []
            
            collection = self.weaviate_service.client.collections.get(
                self.weaviate_service.collection_name
            )
            
            # Create conditions for each related tool
            conditions = []
            for related_name in related_tool_names[:limit]:
                if "." in related_name:
                    conditions.append(
                        Filter.by_property("fullQualifiedName").equal(related_name)
                    )
                else:
                    conditions.append(
                        Filter.by_property("toolName").equal(related_name)
                    )
            
            if not conditions:
                return []
            
            where_filter = Filter.any_of(conditions) if len(conditions) > 1 else conditions[0]
            
            response = collection.query.fetch_objects(
                limit=limit,
                filters=where_filter  # Use filters parameter
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "toolName": obj.properties.get("toolName"),
                    "section": obj.properties.get("section"),
                    "description": obj.properties.get("description"),
                    "actionType": obj.properties.get("actionType"),
                    "resourceType": obj.properties.get("resourceType")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            console.print(f"[red]✗ Failed to get related tools: {e}[/red]")
            raise
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored tools"""
        try:
            collection = self.weaviate_service.client.collections.get(
                self.weaviate_service.collection_name
            )
            
            # Get total count
            total_response = collection.aggregate.over_all(
                total_count=True
            )
            total_count = total_response.total_count
            
            # Get tools by section
            sections_response = collection.aggregate.over_all(
                group_by="section"
            )
            
            section_counts = {}
            if sections_response.groups:
                for group in sections_response.groups:
                    section_name = group.grouped_by.value
                    count = group.total_count
                    section_counts[section_name] = count
            
            # Get tools by action type
            actions_response = collection.aggregate.over_all(
                group_by="actionType"
            )
            
            action_counts = {}
            if actions_response.groups:
                for group in actions_response.groups:
                    action_type = group.grouped_by.value
                    count = group.total_count
                    action_counts[action_type] = count
            
            return {
                "total_tools": total_count,
                "tools_by_section": section_counts,
                "tools_by_action": action_counts,
                "collection_name": self.weaviate_service.collection_name
            }
            
        except Exception as e:
            console.print(f"[red]✗ Failed to get storage stats: {e}[/red]")
            raise

# Example usage and testing functions
def test_enhanced_storage():
    """Test the enhanced storage system"""
    
    # Configuration
    JSON_FILE_PATH = "github_mcp_tools.json"
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Validate configuration
    if not Path(JSON_FILE_PATH).exists():
        console.print(f"[red]✗ JSON file not found: {JSON_FILE_PATH}[/red]")
        return
    
    try:
        # Initialize storage system
        storage = EnhancedGitHubMCPStorage(
            json_file_path=JSON_FILE_PATH,
            weaviate_url=WEAVIATE_URL,
            weaviate_api_key=WEAVIATE_API_KEY,
            embedding_model=EMBEDDING_MODEL
        )
        
        # Initialize storage
        tool_count = storage.initialize_storage()
        console.print(f"[green]✅ Initialized storage with {tool_count} tools[/green]")
        
        # Test search functionality
        console.print("\n[bold]Testing search functionality:[/bold]")
        
        test_queries = [
            "create a new issue",
            "list all pull requests",
            "get workflow status",
            "search repositories",
            "merge pull request"
        ]
        
        for query in test_queries:
            console.print(f"\n[cyan]Query: {query}[/cyan]")
            results = storage.search_tools(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                console.print(f"  {i}. {result['section']}.{result['toolName']}")
                console.print(f"     {result['description'][:100]}...")
                console.print(f"     Score: {result.get('relevance_score', 0):.3f}")
        
        # Test getting storage stats
        console.print("\n[bold]Storage Statistics:[/bold]")
        stats = storage.get_storage_stats()
        console.print(f"Total tools: {stats['total_tools']}")
        console.print(f"Sections: {list(stats['tools_by_section'].keys())}")
        console.print(f"Action types: {list(stats['tools_by_action'].keys())}")
        
        console.print(Panel.fit("🎉 Enhanced Storage Test Complete!", style="bold green"))
        
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}[/red]")
        logger.exception("Test failed")

if __name__ == "__main__":
    test_enhanced_storage()