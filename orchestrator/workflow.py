"""
LangGraph-based orchestrator for the Financial Insights Agent System.
This orchestrator manages the flow between specialized agents using a graph-based approach.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

# LangGraph imports (commented out due to import issues - would be used in production)
# from langgraph import StateGraph, END
# from langgraph.graph import MessageGraph

from core.types import AgentState, TaskType, FinancialKPI, InsightResult, AgentType
from agents.router import RouterAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.recommendation import RecommendationAgent
from agents.visualization import VisualizationAgent
from services.llm import LLMService
from services.vector_store import VectorStoreService
from services.database import DatabaseService

logger = logging.getLogger(__name__)


class FinancialInsightsOrchestrator:
    """
    Main orchestrator that coordinates all specialized agents using a graph-based workflow.
    Implements intelligent routing and context management for optimal agent execution.
    """
    
    def __init__(self, groq_api_key: str, mongodb_uri: str, db_config: Dict):
        # Initialize services
        self.llm_service = LLMService(groq_api_key)
        self.vector_service = VectorStoreService(mongodb_uri)
        self.db_service = DatabaseService(db_config)
        
        # Initialize agents
        self.router_agent = RouterAgent(self.llm_service, self.vector_service)
        self.sql_agent = SQLGeneratorAgent(self.llm_service, self.vector_service, self.db_service)
        self.recommendation_agent = RecommendationAgent(self.llm_service, self.vector_service)
        self.visualization_agent = VisualizationAgent(self.vector_service)
        
        # Agent execution graph (simplified implementation without LangGraph)
        self.execution_graph = self._build_execution_graph()
        
    def _build_execution_graph(self) -> Dict:
        """Build the agent execution graph structure."""
        return {
            'router': {
                'agent': self.router_agent,
                'next_nodes': ['schema_analyzer', 'sql_generator'],
                'conditions': {
                    'simple_query': ['sql_generator'],
                    'complex_analysis': ['schema_analyzer', 'sql_generator', 'visualization'],
                    'dashboard_request': ['schema_analyzer', 'sql_generator', 'visualization', 'recommendation']
                }
            },
            'schema_analyzer': {
                'agent': None,  # Built into database service
                'next_nodes': ['sql_generator'],
                'conditions': {}
            },
            'sql_generator': {
                'agent': self.sql_agent,
                'next_nodes': ['data_executor'],
                'conditions': {}
            },
            'data_executor': {
                'agent': None,  # Built into database service
                'next_nodes': ['visualization', 'recommendation'],
                'conditions': {
                    'visualization_needed': ['visualization'],
                    'recommendations_needed': ['recommendation'],
                    'complete_analysis': ['visualization', 'recommendation']
                }
            },
            'visualization': {
                'agent': self.visualization_agent,
                'next_nodes': ['recommendation'],
                'conditions': {}
            },
            'recommendation': {
                'agent': self.recommendation_agent,
                'next_nodes': ['end'],
                'conditions': {}
            }
        }
    
    async def process_request(self, user_input: str, session_id: str) -> Dict:
        """
        Main entry point for processing user requests through the agent graph.
        """
        start_time = time.time()
        
        # Initialize agent state
        state = AgentState(
            user_input=user_input,
            session_id=session_id
        )
        
        try:
            # Execute the agent workflow
            final_state = await self._execute_workflow(state)
            
            # Create final response
            response = await self._create_final_response(final_state)
            
            # Store insights
            await self._store_session_insights(final_state, session_id)
            
            # Log total execution time
            total_time = time.time() - start_time
            logger.info(f"Request processed in {total_time:.2f}s with confidence {final_state.confidence_score}")
            
            response['execution_metadata'] = {
                'total_execution_time': total_time,
                'agents_executed': final_state.metadata.get('agents_executed', []),
                'confidence_score': final_state.confidence_score,
                'error_count': len(final_state.error_messages)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            return {
                'error': f"System error: {str(e)}",
                'user_input': user_input,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_workflow(self, state: AgentState) -> AgentState:
        """Execute the agent workflow based on the graph structure."""
        current_node = 'router'
        agents_executed = []
        
        while current_node != 'end':
            logger.info(f"Executing agent: {current_node}")
            agents_executed.append(current_node)
            
            # Execute current agent
            if current_node == 'router':
                state = await self.router_agent.route_request(state)
                next_nodes = self._determine_next_nodes(current_node, state)
                
            elif current_node == 'schema_analyzer':
                # Schema analysis is built into database service
                if not state.database_schema:
                    state.database_schema = await self.db_service.get_enhanced_schema()
                next_nodes = ['sql_generator']
                
            elif current_node == 'sql_generator':
                state = await self.sql_agent.generate_sql_queries(state)
                next_nodes = ['data_executor']
                
            elif current_node == 'data_executor':
                state = await self._execute_sql_queries(state)
                next_nodes = self._determine_next_nodes(current_node, state)
                
            elif current_node == 'visualization':
                state = await self.visualization_agent.create_visualizations(state)
                next_nodes = self._determine_next_nodes(current_node, state)
                
            elif current_node == 'recommendation':
                state = await self.recommendation_agent.generate_recommendations(state)
                next_nodes = ['end']
            
            else:
                logger.warning(f"Unknown node: {current_node}")
                next_nodes = ['end']
            
            # Move to next node
            if next_nodes and next_nodes[0] != 'end':
                current_node = next_nodes[0]
            else:
                current_node = 'end'
        
        state.metadata['agents_executed'] = agents_executed
        return state
    
    def _determine_next_nodes(self, current_node: str, state: AgentState) -> List[str]:
        """Determine next nodes based on current state and conditions."""
        node_config = self.execution_graph.get(current_node, {})
        conditions = node_config.get('conditions', {})
        
        if current_node == 'router':
            if state.task_type == TaskType.DASHBOARD_CREATION:
                return ['schema_analyzer']
            elif len(state.target_kpis) > 3:
                return ['schema_analyzer']
            else:
                return ['sql_generator']
        
        elif current_node == 'data_executor':
            next_nodes = []
            
            # Always add visualization for multiple KPIs or dashboard requests
            if len(state.target_kpis) > 1 or state.task_type == TaskType.DASHBOARD_CREATION:
                next_nodes.append('visualization')
            
            # Add recommendations based on task type or if specifically requested
            if (state.task_type in [TaskType.RECOMMENDATION_GENERATION, TaskType.DASHBOARD_CREATION] or
                'recommend' in state.user_input.lower()):
                next_nodes.append('recommendation')
            
            # If no specific next nodes, default to recommendation
            if not next_nodes:
                next_nodes = ['recommendation']
            
            return next_nodes
        
        elif current_node == 'visualization':
            # After visualization, usually go to recommendations
            if state.task_type == TaskType.RECOMMENDATION_GENERATION or len(state.target_kpis) > 1:
                return ['recommendation']
            else:
                return ['end']
        
        # Default to ending the workflow
        return ['end']
    
    async def _execute_sql_queries(self, state: AgentState) -> AgentState:
        """Execute all SQL queries and collect results."""
        try:
            execution_tasks = []
            
            for kpi_name, sql_query in state.sql_queries.items():
                task = self.db_service.execute_query(sql_query)
                execution_tasks.append((kpi_name, task))
            
            # Execute queries in parallel
            for kpi_name, task in execution_tasks:
                try:
                    result = await task
                    state.query_results[kpi_name] = result
                except Exception as e:
                    logger.error(f"Query execution failed for {kpi_name}: {e}")
                    state.error_messages.append(f"Query failed for {kpi_name}: {str(e)}")
                    state.query_results[kpi_name] = {"error": str(e)}
            
            logger.info(f"Executed {len(state.sql_queries)} queries, {len(state.query_results)} results")
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            state.error_messages.append(f"SQL execution failed: {str(e)}")
        
        return state
    
    async def _create_final_response(self, state: AgentState) -> Dict:
        """Create the final response from agent state."""
        response = {
            'user_input': state.user_input,
            'session_id': state.session_id,
            'task_type': state.task_type.value if state.task_type else 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add query results
        if state.query_results:
            response['kpi_data'] = state.query_results
        
        # Add visualizations
        if state.visualizations:
            response['visualizations'] = state.visualizations
        
        # Add recommendations
        if state.recommendations:
            response['recommendations'] = state.recommendations
        
        # Add metadata
        response['metadata'] = state.metadata
        response['confidence_score'] = state.confidence_score
        
        # Add errors if any
        if state.error_messages:
            response['errors'] = state.error_messages
        
        return response
    
    async def _store_session_insights(self, state: AgentState, session_id: str):
        """Store insights and session state for future reference."""
        try:
            # Create insight result
            insight = InsightResult(
                insight_id=str(uuid.uuid4()),
                insight_type="financial_analysis",
                content={
                    'query_results': state.query_results,
                    'recommendations': state.recommendations,
                    'task_type': state.task_type.value if state.task_type else None
                },
                metadata=state.metadata,
                confidence_score=state.confidence_score,
                timestamp=datetime.now(),
                agent_path=state.metadata.get('agents_executed', [])
            )
            
            # Store insight
            await self.vector_service.store_insight(insight, session_id)
            
            # Store agent state
            await self.vector_service.store_agent_state(state)
            
        except Exception as e:
            logger.error(f"Failed to store session insights: {e}")
    
    async def get_session_analytics(self, session_id: str) -> Dict:
        """Get analytics for a specific session."""
        try:
            analytics = await self.vector_service.get_session_analytics(session_id)
            usage_stats = self.llm_service.get_usage_stats()
            
            return {
                'session_analytics': analytics,
                'model_usage': usage_stats,
                'cost_analysis': self._calculate_session_costs(usage_stats)
            }
        except Exception as e:
            logger.error(f"Failed to get session analytics: {e}")
            return {}
    
    def _calculate_session_costs(self, usage_stats: Dict) -> Dict:
        """Calculate costs for the session."""
        total_cost = usage_stats.get('total', {}).get('total_cost', 0)
        
        return {
            'total_estimated_cost': total_cost,
            'cost_by_agent': {
                agent: stats.get('total_cost', 0)
                for agent, stats in usage_stats.items()
                if agent != 'total'
            },
            'cost_optimization_suggestions': self._get_cost_optimization_suggestions(usage_stats)
        }
    
    def _get_cost_optimization_suggestions(self, usage_stats: Dict) -> List[str]:
        """Generate cost optimization suggestions."""
        suggestions = []
        
        # Check for high token usage agents
        for agent, stats in usage_stats.items():
            if agent == 'total':
                continue
                
            avg_tokens = stats.get('avg_tokens_per_call', 0)
            if avg_tokens > 1000:
                suggestions.append(f"Consider optimizing prompts for {agent} agent (avg {avg_tokens} tokens per call)")
        
        # Check total costs
        total_cost = usage_stats.get('total', {}).get('total_cost', 0)
        if total_cost > 1.0:
            suggestions.append("Consider using smaller models for routine tasks to reduce costs")
        
        return suggestions
