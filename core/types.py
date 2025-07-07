"""
Core types and enums for the Financial Insights Agent System
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


class AgentType(Enum):
    """Defines the types of specialized agents in the system."""
    ROUTER = "router"
    SCHEMA_ANALYZER = "schema_analyzer"
    SQL_GENERATOR = "sql_generator"
    DATA_EXECUTOR = "data_executor"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    CONTEXT_MANAGER = "context_manager"


class TaskType(Enum):
    """Types of tasks that can be performed."""
    KPI_ANALYSIS = "kpi_analysis"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CUSTOM_QUERY = "custom_query"
    DASHBOARD_CREATION = "dashboard_creation"
    RECOMMENDATION_GENERATION = "recommendation_generation"


class FinancialKPI(Enum):
    """Financial KPIs with business context for few-shot prompting."""
    GROSS_TRANSACTION_VALUE = {
        "name": "Gross Transaction Value (GTV)",
        "description": "Total monetary value of all successful transactions",
        "calculation": "SUM(transaction_amount) WHERE status = 'successful'",
        "example_sql": "SELECT SUM(amount) as gtv FROM transactions WHERE status = 'completed' AND created_at >= '2024-01-01'"
    }
    TRANSACTION_VOLUME = {
        "name": "Transaction Volume",
        "description": "Total number of successful transactions",
        "calculation": "COUNT(*) WHERE status = 'successful'",
        "example_sql": "SELECT COUNT(*) as volume FROM transactions WHERE status = 'completed'"
    }
    APPROVAL_RATE = {
        "name": "Approval Rate",
        "description": "Percentage of transactions successfully processed",
        "calculation": "(successful_transactions / total_transactions) * 100",
        "example_sql": "SELECT (COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*)) as approval_rate FROM transactions"
    }
    FRAUD_RATE = {
        "name": "Fraud Rate",
        "description": "Percentage of transactions identified as fraudulent",
        "calculation": "(fraudulent_transactions / total_transactions) * 100",
        "example_sql": "SELECT (COUNT(CASE WHEN is_fraud = true THEN 1 END) * 100.0 / COUNT(*)) as fraud_rate FROM transactions"
    }
    AVERAGE_TRANSACTION_VALUE = {
        "name": "Average Transaction Value (ATV)",
        "description": "Average monetary value per transaction",
        "calculation": "AVG(transaction_amount) WHERE status = 'successful'",
        "example_sql": "SELECT AVG(amount) as atv FROM transactions WHERE status = 'completed'"
    }
    ACTIVE_USERS = {
        "name": "Active Users",
        "description": "Number of unique users with at least one transaction",
        "calculation": "COUNT(DISTINCT user_id) WHERE has_transaction = true",
        "example_sql": "SELECT COUNT(DISTINCT user_id) as active_users FROM transactions WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'"
    }


@dataclass
class AgentState:
    """Shared state passed between agents in the graph."""
    user_input: str
    session_id: str
    task_type: Optional[TaskType] = None
    target_kpis: List[FinancialKPI] = field(default_factory=list)
    database_schema: Optional[Dict] = None
    sql_queries: Dict[str, str] = field(default_factory=dict)
    query_results: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, str] = field(default_factory=dict)
    recommendations: Optional[Dict] = None
    error_messages: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    

@dataclass
class InsightResult:
    """Enhanced insight result with agent tracking."""
    insight_id: str
    insight_type: str
    content: Dict
    metadata: Dict
    confidence_score: float
    timestamp: datetime
    agent_path: List[str] = field(default_factory=list)
    model_used: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for different models used by agents."""
    model_name: str
    temperature: float
    max_tokens: int
    use_case: str
    cost_per_token: float = 0.0
