"""
Enhanced vector store and session management services.
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from core.types import InsightResult, AgentState

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Enhanced vector store with better retrieval and context management."""
    
    def __init__(self, mongodb_uri: str, collection_name: str = "financial_insights"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_db
        self.collection = self.db[collection_name]
        self.sessions_collection = self.db.sessions
        self.agent_logs_collection = self.db.agent_logs
        
        # Initialize embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB with proper settings for version 0.4.x
        try:
            # For ChromaDB 0.4.x, use PersistentClient without proxies parameter
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        except Exception as e:
            logger.warning(f"Failed to create PersistentClient: {e}")
            try:
                # Fallback to EphemeralClient for development/testing
                self.chroma_client = chromadb.EphemeralClient()
                logger.info("Using EphemeralClient as fallback")
            except Exception as e2:
                logger.error(f"Failed to create any ChromaDB client: {e2}")
                # Create a mock client that won't break the application
                self.chroma_client = self._create_mock_client()
        
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="financial_insights_enhanced"
            )
            logger.info("ChromaDB collection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            # The mock client will handle this gracefully
    
    async def store_insight(self, insight: InsightResult, session_id: str):
        """Store insight with enhanced metadata and embedding."""
        try:
            # Create content text for embedding
            content_text = self._create_searchable_text(insight.content)
            embedding = self.encoder.encode(content_text).tolist()
            
            # Prepare document
            doc = {
                'insight_id': insight.insight_id,
                'insight_type': insight.insight_type,
                'content': insight.content,
                'metadata': insight.metadata,
                'confidence_score': insight.confidence_score,
                'timestamp': insight.timestamp,
                'session_id': session_id,
                'agent_path': insight.agent_path,
                'model_used': insight.model_used,
                'embedding': embedding,
                'searchable_text': content_text
            }
            
            # Store in MongoDB
            await asyncio.to_thread(self.collection.insert_one, doc)
            
            # Store in ChromaDB (with error handling)
            try:
                self.chroma_collection.add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[{
                        'insight_id': insight.insight_id,
                        'session_id': session_id,
                        'type': insight.insight_type,
                        'confidence': insight.confidence_score,
                        'timestamp': insight.timestamp.isoformat()
                    }],
                    ids=[insight.insight_id]
                )
            except Exception as chroma_error:
                logger.warning(f"Failed to store in ChromaDB: {chroma_error}")
            
            logger.info(f"Stored insight {insight.insight_id} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            raise
    
    async def retrieve_relevant_context(self, query: str, session_id: str, 
                                      limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Retrieve relevant context using hybrid search."""
        try:
            query_embedding = self.encoder.encode(query).tolist()
            
            # Search in ChromaDB (with error handling)
            relevant_results = []
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where={"session_id": session_id}
                )
                
                # Filter by similarity threshold
                if results['documents']:
                    for i, doc in enumerate(results['documents'][0]):
                        distance = results['distances'][0][i] if 'distances' in results else 0.5
                        similarity = 1 - distance  # Convert distance to similarity
                        
                        if similarity >= similarity_threshold:
                            metadata = results['metadatas'][0][i]
                            relevant_results.append({
                                'content': doc,
                                'metadata': metadata,
                                'similarity_score': similarity
                            })
            except Exception as chroma_error:
                logger.warning(f"ChromaDB query failed: {chroma_error}")
                # Continue without vector search results
            
            # Also get recent insights from the same session
            recent_insights = await self._get_recent_session_insights(session_id, limit=3)
            
            return {
                'similar_insights': relevant_results,
                'recent_insights': recent_insights
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return {'similar_insights': [], 'recent_insights': []}
    
    async def _get_recent_session_insights(self, session_id: str, limit: int = 3) -> List[Dict]:
        """Get recent insights from the same session."""
        try:
            cursor = self.collection.find(
                {'session_id': session_id},
                {'content': 1, 'metadata': 1, 'timestamp': 1, 'insight_type': 1}
            ).sort('timestamp', -1).limit(limit)
            
            return await asyncio.to_thread(list, cursor)
        except Exception as e:
            logger.error(f"Failed to get recent insights: {e}")
            return []
    
    async def store_agent_state(self, state: AgentState):
        """Store agent state for session continuity."""
        try:
            doc = {
                'session_id': state.session_id,
                'user_input': state.user_input,
                'task_type': state.task_type.value if state.task_type else None,
                'target_kpis': [kpi.name for kpi in state.target_kpis],
                'query_results': state.query_results,
                'confidence_score': state.confidence_score,
                'metadata': state.metadata,
                'conversation_history': state.conversation_history,
                'timestamp': datetime.now(),
                'error_messages': state.error_messages
            }
            
            await asyncio.to_thread(
                self.sessions_collection.replace_one,
                {'session_id': state.session_id},
                doc,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Failed to store agent state: {e}")
            raise
    
    async def get_agent_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve agent state for session."""
        try:
            doc = await asyncio.to_thread(
                self.sessions_collection.find_one,
                {'session_id': session_id}
            )
            return doc
        except Exception as e:
            logger.error(f"Failed to get agent state: {e}")
            return None
    
    async def log_agent_execution(self, session_id: str, agent_name: str, 
                                 input_data: Dict, output_data: Dict, 
                                 execution_time: float, model_used: str):
        """Log agent execution for analysis and debugging."""
        try:
            log_entry = {
                'session_id': session_id,
                'agent_name': agent_name,
                'input_data': input_data,
                'output_data': output_data,
                'execution_time': execution_time,
                'model_used': model_used,
                'timestamp': datetime.now()
            }
            
            await asyncio.to_thread(self.agent_logs_collection.insert_one, log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log agent execution: {e}")
    
    def _create_searchable_text(self, content: Dict) -> str:
        """Create searchable text from content dictionary."""
        def extract_text(obj, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return ""
            
            if isinstance(obj, str):
                return obj
            elif isinstance(obj, (int, float)):
                return str(obj)
            elif isinstance(obj, dict):
                texts = []
                for key, value in obj.items():
                    texts.append(f"{key}: {extract_text(value, max_depth, current_depth + 1)}")
                return " ".join(texts)
            elif isinstance(obj, list):
                return " ".join([extract_text(item, max_depth, current_depth + 1) for item in obj])
            else:
                return str(obj)
        
        return extract_text(content)
    
    async def get_session_analytics(self, session_id: str) -> Dict:
        """Get analytics for a specific session."""
        try:
            # Get session insights
            insights_cursor = self.collection.find({'session_id': session_id})
            insights = await asyncio.to_thread(list, insights_cursor)
            
            # Get agent logs
            logs_cursor = self.agent_logs_collection.find({'session_id': session_id})
            logs = await asyncio.to_thread(list, logs_cursor)
            
            # Calculate analytics
            analytics = {
                'total_insights': len(insights),
                'total_agent_executions': len(logs),
                'avg_confidence_score': sum(i.get('confidence_score', 0) for i in insights) / len(insights) if insights else 0,
                'most_used_agents': {},
                'execution_times': [],
                'error_count': len([i for i in insights if 'error' in i.get('content', {})])
            }
            
            # Agent usage statistics
            for log in logs:
                agent_name = log.get('agent_name', 'unknown')
                analytics['most_used_agents'][agent_name] = analytics['most_used_agents'].get(agent_name, 0) + 1
                analytics['execution_times'].append(log.get('execution_time', 0))
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get session analytics: {e}")
            return {}
    
    def _create_mock_client(self):
        """Create a mock ChromaDB client that doesn't break the application."""
        class MockCollection:
            def add(self, **kwargs):
                logger.warning("MockCollection.add called - ChromaDB not available")
                pass
            
            def query(self, **kwargs):
                logger.warning("MockCollection.query called - ChromaDB not available")
                return {'documents': [], 'metadatas': [], 'distances': []}
        
        class MockClient:
            def get_or_create_collection(self, **kwargs):
                return MockCollection()
        
        logger.warning("Using mock ChromaDB client - vector search will be disabled")
        return MockClient()
