"""
Core types and configurations for the Financial Insights Agent System.
"""

from .types import (
    AgentType, TaskType, FinancialKPI, AgentState, 
    InsightResult, ModelConfig
)
from .models import ModelManager
from .prompts import PromptTemplates

__all__ = [
    'AgentType',
    'TaskType', 
    'FinancialKPI',
    'AgentState',
    'InsightResult',
    'ModelConfig',
    'ModelManager',
    'PromptTemplates'
]
