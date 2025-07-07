"""
Router Agent - Intelligent request routing and task classification.
"""
import asyncio
import logging
import time
from typing import Dict, Any
from core.types import AgentState, TaskType, FinancialKPI
from core.prompts import PromptTemplates
from services.llm import LLMService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    Intelligent router that analyzes user input and determines the optimal
    execution path through the agent system.
    """
    
    def __init__(self, llm_service: LLMService, vector_service: VectorStoreService):
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.agent_type = "router"
    
    async def route_request(self, state: AgentState) -> AgentState:
        """Analyze user input and determine routing strategy."""
        start_time = time.time()
        
        try:
            # Get relevant context from previous interactions
            context = await self.vector_service.retrieve_relevant_context(
                state.user_input, state.session_id, limit=3
            )
            
            # Generate routing decision using few-shot prompting
            prompt = PromptTemplates.get_router_prompt(state.user_input)
            
            # Add context if available
            if context['similar_insights']:
                context_text = "\n".join([
                    f"Previous insight: {insight['content'][:200]}..."
                    for insight in context['similar_insights'][:2]
                ])
                prompt += f"\n\nRelevant context from previous interactions:\n{context_text}"
            
            result = await self.llm_service.generate_structured_response(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                agent_type=self.agent_type
            )
            
            # Parse routing decision
            routing_decision = result['result']
            
            if 'error' in routing_decision:
                state.error_messages.append(f"Router error: {routing_decision['error']}")
                state.task_type = TaskType.CUSTOM_QUERY  # Default fallback
            else:
                # Update state with routing decision
                state.task_type = TaskType(routing_decision.get('task_type', 'CUSTOM_QUERY'))
                
                # Map KPI names to enum values
                kpi_names = routing_decision.get('target_kpis', [])
                state.target_kpis = []
                for kpi_name in kpi_names:
                    for kpi in FinancialKPI:
                        if kpi.name == kpi_name:
                            state.target_kpis.append(kpi)
                            break
                
                state.confidence_score = routing_decision.get('confidence', 0.0)
                
                # Add metadata
                state.metadata.update({
                    'routing_decision': routing_decision,
                    'context_used': len(context['similar_insights']),
                    'time_filter': routing_decision.get('time_filter'),
                    'model_used': result['metadata']['model_used']
                })
            
            # Log agent execution
            execution_time = time.time() - start_time
            await self.vector_service.log_agent_execution(
                session_id=state.session_id,
                agent_name=self.agent_type,
                input_data={'user_input': state.user_input},
                output_data=routing_decision,
                execution_time=execution_time,
                model_used=result['metadata']['model_used']
            )
            
            logger.info(f"Routed request to task: {state.task_type.value}, "
                       f"KPIs: {[kpi.name for kpi in state.target_kpis]}, "
                       f"Confidence: {state.confidence_score}")
            
        except Exception as e:
            logger.error(f"Router agent failed: {e}")
            state.error_messages.append(f"Routing failed: {str(e)}")
            state.task_type = TaskType.CUSTOM_QUERY
        
        return state
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the router agent."""
        return """
        You are an intelligent router for a financial analytics system. Your role is to:
        
        1. Analyze user input to understand their intent
        2. Determine the most appropriate task type
        3. Identify relevant financial KPIs mentioned or implied
        4. Extract time-based filters if mentioned
        5. Assess confidence in your routing decision
        
        Be precise and consider the business context. If the user mentions multiple intents,
        prioritize based on the primary action requested.
        
        Always return a valid JSON object with the required fields:
        - task_type: One of [KPI_ANALYSIS, TREND_ANALYSIS, COMPARATIVE_ANALYSIS, CUSTOM_QUERY, DASHBOARD_CREATION, RECOMMENDATION_GENERATION]
        - target_kpis: List of relevant KPI names
        - confidence: Float between 0 and 1
        - time_filter: Optional time-based filter mentioned
        """
    
    async def should_route_to_specialized_path(self, state: AgentState) -> bool:
        """Determine if request needs specialized routing."""
        # Check for complex multi-step requests
        complex_keywords = [
            'compare', 'trend', 'forecast', 'predict', 'analyze correlation',
            'dashboard', 'report', 'deep dive', 'investigation'
        ]
        
        user_input_lower = state.user_input.lower()
        return any(keyword in user_input_lower for keyword in complex_keywords) or \
               len(state.target_kpis) > 3 or \
               state.task_type in [TaskType.TREND_ANALYSIS, TaskType.COMPARATIVE_ANALYSIS, TaskType.DASHBOARD_CREATION]
    
    def get_execution_plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate execution plan based on routing decision."""
        plan = {
            'primary_path': state.task_type.value,
            'agents_required': [],
            'parallel_execution': False,
            'estimated_complexity': 'low'
        }
        
        # Determine required agents based on task type
        if state.task_type == TaskType.KPI_ANALYSIS:
            plan['agents_required'] = ['schema_analyzer', 'sql_generator', 'data_executor']
            if len(state.target_kpis) > 1:
                plan['parallel_execution'] = True
        
        elif state.task_type == TaskType.DASHBOARD_CREATION:
            plan['agents_required'] = [
                'schema_analyzer', 'sql_generator', 'data_executor', 
                'visualization', 'recommendation'
            ]
            plan['estimated_complexity'] = 'high'
        
        elif state.task_type == TaskType.TREND_ANALYSIS:
            plan['agents_required'] = [
                'schema_analyzer', 'sql_generator', 'data_executor', 
                'visualization', 'recommendation'
            ]
            plan['estimated_complexity'] = 'medium'
        
        elif state.task_type == TaskType.RECOMMENDATION_GENERATION:
            plan['agents_required'] = ['recommendation']
            if not state.query_results:
                plan['agents_required'] = [
                    'schema_analyzer', 'sql_generator', 'data_executor', 'recommendation'
                ]
        
        else:  # CUSTOM_QUERY or others
            plan['agents_required'] = ['schema_analyzer', 'sql_generator', 'data_executor']
        
        # Add visualization if not already included and multiple KPIs
        if len(state.target_kpis) > 1 and 'visualization' not in plan['agents_required']:
            plan['agents_required'].append('visualization')
        
        return plan
