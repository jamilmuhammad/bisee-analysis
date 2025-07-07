"""
Enhanced LLM service with model-specific configurations and optimization.
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from groq import Groq

# Try to import LangChain components with fallback
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class ChatGroq:
        def __init__(self, **kwargs):
            pass
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    class SystemMessage:
        def __init__(self, content):
            self.content = content
    class PromptTemplate:
        @staticmethod
        def from_template(template):
            return template

from core.models import ModelManager

logger = logging.getLogger(__name__)


class LLMService:
    """Enhanced LLM service with model optimization and token management."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize Groq client with minimal parameters to avoid compatibility issues
        try:
            # Try with just the API key first
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        except TypeError as e:
            if "proxies" in str(e):
                logger.warning(f"Groq client initialization failed due to proxies parameter: {e}")
                # Try to import and create with minimal httpx client
                try:
                    import httpx
                    # Create a custom httpx client without the proxies parameter
                    custom_client = httpx.Client(timeout=30.0)
                    self.groq_client = Groq(api_key=api_key, http_client=custom_client)
                    logger.info("Groq client initialized with custom httpx client")
                except Exception as e2:
                    logger.error(f"Failed to initialize Groq client with custom client: {e2}")
                    self.groq_client = None
            else:
                logger.error(f"Groq client initialization failed with TypeError: {e}")
                self.groq_client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Groq client: {e}")
            self.groq_client = None
        
        self.model_configs = ModelManager.MODELS
        self._token_usage = {}
        self._cost_tracking = {}
    
    def get_optimized_llm(self, agent_type: str):
        """Get optimized LLM instance for specific agent type."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using direct Groq client")
            return None
            
        config = ModelManager.get_model_config(agent_type)
        
        try:
            return ChatGroq(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                groq_api_key=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to create ChatGroq instance: {e}")
            return None
    
    async def generate_structured_response(self, prompt: str, system_prompt: str, 
                                         agent_type: str, response_format: str = "json") -> Dict:
        """Generate structured response with model optimization."""
        config = ModelManager.get_model_config(agent_type)
        start_time = time.time()
        
        # Check if Groq client is available
        if not self.groq_client:
            logger.error("Groq client not available")
            return {
                "error": "LLM service not available",
                "agent_type": agent_type,
                "timestamp": time.time()
            }
        
        try:
            # Track token usage
            prompt_tokens = self._estimate_tokens(f"{system_prompt}\n{prompt}")
            
            if response_format == "json":
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
            else:
                # Use LangChain for non-JSON responses if available
                if LANGCHAIN_AVAILABLE:
                    llm = self.get_optimized_llm(agent_type)
                    if llm:
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=prompt)
                        ]
                        response = await llm.ainvoke(messages)
                        result = {"response": response.content}
                    else:
                        # Fallback to direct Groq call
                        response = await asyncio.to_thread(
                            self.groq_client.chat.completions.create,
                            model=config.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=config.temperature,
                            max_tokens=config.max_tokens
                        )
                        result = {"response": response.choices[0].message.content}
                else:
                    # Direct Groq call without LangChain
                    response = await asyncio.to_thread(
                        self.groq_client.chat.completions.create,
                        model=config.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    )
                    result = {"response": response.choices[0].message.content}
            
            # Track usage and costs
            execution_time = time.time() - start_time
            completion_tokens = self._estimate_tokens(str(result))
            total_tokens = prompt_tokens + completion_tokens
            
            self._update_usage_tracking(agent_type, total_tokens, execution_time, config.cost_per_token)
            
            return {
                "result": result,
                "metadata": {
                    "model_used": config.model_name,
                    "tokens_used": total_tokens,
                    "execution_time": execution_time,
                    "estimated_cost": total_tokens * config.cost_per_token
                }
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {agent_type}: {e}")
            return {
                "result": {"error": "Failed to parse LLM response as JSON"},
                "metadata": {"error": str(e)}
            }
        except Exception as e:
            logger.error(f"LLM generation failed for {agent_type}: {e}")
            return {
                "result": {"error": f"LLM generation failed: {str(e)}"},
                "metadata": {"error": str(e)}
            }
    
    async def generate_with_langchain(self, template: str, variables: Dict, agent_type: str) -> Dict:
        """Generate response using LangChain prompt templates."""
        if not LANGCHAIN_AVAILABLE:
            # Fallback to direct Groq client
            return await self.generate_structured_response(
                prompt=template.format(**variables),
                system_prompt="You are a helpful AI assistant.",
                agent_type=agent_type,
                response_format="text"
            )
            
        try:
            llm = self.get_optimized_llm(agent_type)
            if llm is None:
                # Fallback to direct Groq client
                return await self.generate_structured_response(
                    prompt=template.format(**variables),
                    system_prompt="You are a helpful AI assistant.",
                    agent_type=agent_type,
                    response_format="text"
                )
                
            prompt_template = PromptTemplate.from_template(template)
            
            start_time = time.time()
            
            # Create prompt
            formatted_prompt = prompt_template.format(**variables)
            
            # Generate response
            response = await llm.ainvoke([HumanMessage(content=formatted_prompt)])
            
            execution_time = time.time() - start_time
            
            return {
                "result": {"response": response.content},
                "metadata": {
                    "model_used": self.model_configs[agent_type].model_name,
                    "execution_time": execution_time,
                    "prompt_template": template[:100] + "..." if len(template) > 100 else template
                }
            }
            
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            # Fallback to direct Groq client
            return await self.generate_structured_response(
                prompt=template.format(**variables),
                system_prompt="You are a helpful AI assistant.",
                agent_type=agent_type,
                response_format="text"
            )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _update_usage_tracking(self, agent_type: str, tokens: int, execution_time: float, cost_per_token: float):
        """Update usage and cost tracking."""
        if agent_type not in self._token_usage:
            self._token_usage[agent_type] = {'tokens': 0, 'calls': 0, 'total_time': 0}
            self._cost_tracking[agent_type] = 0
        
        self._token_usage[agent_type]['tokens'] += tokens
        self._token_usage[agent_type]['calls'] += 1
        self._token_usage[agent_type]['total_time'] += execution_time
        self._cost_tracking[agent_type] += tokens * cost_per_token
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics for all agent types."""
        stats = {}
        for agent_type in self._token_usage:
            usage = self._token_usage[agent_type]
            stats[agent_type] = {
                'total_tokens': usage['tokens'],
                'total_calls': usage['calls'],
                'total_time': usage['total_time'],
                'avg_tokens_per_call': usage['tokens'] / usage['calls'] if usage['calls'] > 0 else 0,
                'avg_time_per_call': usage['total_time'] / usage['calls'] if usage['calls'] > 0 else 0,
                'total_cost': self._cost_tracking.get(agent_type, 0),
                'model_name': self.model_configs.get(agent_type, {}).model_name
            }
        
        # Calculate totals
        stats['total'] = {
            'total_tokens': sum(usage['tokens'] for usage in self._token_usage.values()),
            'total_calls': sum(usage['calls'] for usage in self._token_usage.values()),
            'total_time': sum(usage['total_time'] for usage in self._token_usage.values()),
            'total_cost': sum(self._cost_tracking.values())
        }
        
        return stats
    
    async def batch_generate(self, requests: List[Dict], agent_type: str) -> List[Dict]:
        """Generate multiple responses in batch for efficiency."""
        tasks = []
        
        for request in requests:
            task = self.generate_structured_response(
                prompt=request['prompt'],
                system_prompt=request['system_prompt'],
                agent_type=agent_type,
                response_format=request.get('response_format', 'json')
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "result": {"error": str(result)},
                        "metadata": {"error": str(result), "request_index": i}
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [{"result": {"error": str(e)}, "metadata": {"error": str(e)}} for _ in requests]
