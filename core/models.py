"""
Model configurations for different agents to optimize token usage and performance.
"""
from core.types import ModelConfig


class ModelManager:
    """Manages different model configurations for various agent tasks."""
    
    MODELS = {
        # Fast, cost-effective model for routing and simple tasks
        "router": ModelConfig(
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=512,
            use_case="Intent classification and routing",
            cost_per_token=0.00005
        ),
        
        # Precise model for SQL generation requiring accuracy
        "sql_generator": ModelConfig(
            model_name="llama3-70b-8192",
            temperature=0.0,
            max_tokens=1024,
            use_case="SQL query generation and optimization",
            cost_per_token=0.0002
        ),
        
        # Creative model for recommendations and insights
        "recommendation": ModelConfig(
            model_name="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=2048,
            use_case="Strategic recommendations and insights",
            cost_per_token=0.00015
        ),
        
        # Balanced model for schema analysis
        "schema_analyzer": ModelConfig(
            model_name="llama3-8b-8192",
            temperature=0.2,
            max_tokens=1024,
            use_case="Database schema analysis and understanding",
            cost_per_token=0.00005
        ),
        
        # Context-aware model for conversation management
        "context_manager": ModelConfig(
            model_name="llama3-8b-8192",
            temperature=0.2,
            max_tokens=512,
            use_case="Context extraction and session management",
            cost_per_token=0.00005
        )
    }
    
    @classmethod
    def get_model_config(cls, agent_type: str) -> ModelConfig:
        """Get model configuration for specific agent type."""
        return cls.MODELS.get(agent_type, cls.MODELS["router"])
    
    @classmethod
    def estimate_cost(cls, agent_type: str, token_count: int) -> float:
        """Estimate cost for using specific model."""
        config = cls.get_model_config(agent_type)
        return config.cost_per_token * token_count
