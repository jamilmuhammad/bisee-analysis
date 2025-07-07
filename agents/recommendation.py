"""
Recommendation Agent - Strategic insights and business recommendations.
"""
import asyncio
import logging
import time
from typing import Dict, List, Any
from core.types import AgentState
from core.prompts import PromptTemplates
from services.llm import LLMService
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """
    Agent specialized in generating strategic recommendations using
    Descriptive, Predictive, and Prescriptive analytics framework.
    """
    
    def __init__(self, llm_service: LLMService, vector_service: VectorStoreService):
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.agent_type = "recommendation"
    
    async def generate_recommendations(self, state: AgentState) -> AgentState:
        """Generate strategic recommendations based on data analysis."""
        start_time = time.time()
        
        try:
            # Ensure we have query results to base recommendations on
            if not state.query_results:
                state.error_messages.append("No data available for recommendation generation")
                return state
            
            # Get historical context for better recommendations
            historical_context = await self.vector_service.retrieve_relevant_context(
                query="recommendations insights analysis",
                session_id=state.session_id,
                limit=3
            )
            
            # Generate recommendations using few-shot prompting
            prompt = PromptTemplates.get_recommendation_prompt(
                state.query_results,
                state.database_schema
            )
            
            # Add historical context if available
            if historical_context['similar_insights']:
                context_text = "\n".join([
                    f"Previous recommendation: {insight['content'][:150]}..."
                    for insight in historical_context['similar_insights'][:2]
                ])
                prompt += f"\n\nHistorical Context:\n{context_text}"
            
            result = await self.llm_service.generate_structured_response(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                agent_type=self.agent_type
            )
            
            recommendations = result['result']
            
            if 'error' not in recommendations:
                state.recommendations = recommendations
                
                # Add metadata
                state.metadata['recommendation_metadata'] = {
                    'model_used': result['metadata']['model_used'],
                    'historical_context_used': len(historical_context['similar_insights']),
                    'confidence_factors': self._assess_recommendation_confidence(
                        recommendations, state.query_results
                    )
                }
                
                # Update confidence score based on data quality
                data_quality_score = self._assess_data_quality(state.query_results)
                state.confidence_score = min(
                    state.confidence_score + data_quality_score * 0.2, 1.0
                )
            else:
                state.error_messages.append(f"Recommendation generation failed: {recommendations['error']}")
            
            # Log execution
            execution_time = time.time() - start_time
            await self.vector_service.log_agent_execution(
                session_id=state.session_id,
                agent_name=self.agent_type,
                input_data={'data_points': len(state.query_results)},
                output_data={'recommendations_generated': 'descriptive' in recommendations},
                execution_time=execution_time,
                model_used=result['metadata']['model_used']
            )
            
            logger.info(f"Generated recommendations in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Recommendation agent failed: {e}")
            state.error_messages.append(f"Recommendation generation failed: {str(e)}")
        
        return state
    
    def _assess_recommendation_confidence(self, recommendations: Dict, query_results: Dict) -> Dict:
        """Assess confidence factors for recommendations."""
        factors = {
            'data_completeness': 0.0,
            'trend_clarity': 0.0,
            'sample_size': 0.0,
            'actionability': 0.0
        }
        
        # Assess data completeness
        total_data_points = sum(
            len(result) if isinstance(result, list) else 1
            for result in query_results.values()
            if not isinstance(result, dict) or 'error' not in result
        )
        
        if total_data_points > 100:
            factors['data_completeness'] = 1.0
        elif total_data_points > 10:
            factors['data_completeness'] = 0.7
        else:
            factors['data_completeness'] = 0.3
        
        # Assess actionability of recommendations
        if 'prescriptive' in recommendations and recommendations['prescriptive']:
            prescriptive_items = recommendations['prescriptive']
            if isinstance(prescriptive_items, list) and len(prescriptive_items) >= 3:
                factors['actionability'] = 1.0
            elif isinstance(prescriptive_items, list) and len(prescriptive_items) >= 1:
                factors['actionability'] = 0.7
            else:
                factors['actionability'] = 0.3
        
        return factors
    
    def _assess_data_quality(self, query_results: Dict) -> float:
        """Assess the quality of data for recommendations."""
        quality_score = 0.0
        total_metrics = 0
        
        for kpi_name, result in query_results.items():
            total_metrics += 1
            
            if isinstance(result, dict) and 'error' in result:
                continue
            
            if isinstance(result, (int, float)) and result > 0:
                quality_score += 1.0
            elif isinstance(result, list) and len(result) > 0:
                quality_score += 1.0
            else:
                quality_score += 0.3
        
        return quality_score / total_metrics if total_metrics > 0 else 0.0
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for recommendation generation."""
        return """
        You are a senior financial strategy consultant with deep expertise in fintech and e-wallet businesses.
        Your role is to provide actionable, data-driven recommendations using a structured analytical framework.
        
        Framework:
        1. **Descriptive Analytics**: What happened? Summarize key findings from the data objectively.
        2. **Predictive Analytics**: What will likely happen? Project trends and identify potential scenarios.
        3. **Prescriptive Analytics**: What should we do? Provide specific, actionable recommendations.
        
        Guidelines:
        - Base all insights on the provided data
        - Quantify impact where possible
        - Prioritize recommendations by business impact
        - Consider implementation feasibility
        - Include risk assessment for recommendations
        - Provide timeline estimates for implementation
        
        Output Structure:
        - descriptive: Key findings and current state analysis
        - predictive: Trend analysis and future projections
        - prescriptive: Specific action items with priority levels
        - priority: Overall priority level (high/medium/low)
        - impact: Estimated business impact
        - risks: Potential risks and mitigation strategies
        - timeline: Implementation timeline recommendations
        """
    
    async def generate_executive_summary(self, state: AgentState) -> Dict:
        """Generate an executive summary of insights and recommendations."""
        if not state.recommendations:
            return {"error": "No recommendations available for executive summary"}
        
        summary_prompt = f"""
        Create a concise executive summary based on the following analysis:
        
        Data Results: {state.query_results}
        Recommendations: {state.recommendations}
        
        The summary should be:
        - Maximum 3 paragraphs
        - Focus on key business insights
        - Highlight critical action items
        - Include quantified impact where available
        
        Return JSON with: summary, key_metrics, top_actions, business_impact
        """
        
        result = await self.llm_service.generate_structured_response(
            prompt=summary_prompt,
            system_prompt="You are an executive communication specialist for financial services.",
            agent_type=self.agent_type
        )
        
        return result['result']
