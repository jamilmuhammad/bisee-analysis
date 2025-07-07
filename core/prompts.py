"""
Few-shot prompting templates for enhanced agent performance.
"""

from typing import Dict, List
from core.types import FinancialKPI


class PromptTemplates:
    """Contains few-shot prompting templates for different agents."""
    
    ROUTER_EXAMPLES = [
        {
            "input": "Show me the total transaction volume for last month",
            "output": {
                "task_type": "KPI_ANALYSIS",
                "target_kpis": ["TRANSACTION_VOLUME"],
                "time_filter": "last_month",
                "confidence": 0.95
            }
        },
        {
            "input": "I want to see fraud trends compared to last quarter",
            "output": {
                "task_type": "TREND_ANALYSIS",
                "target_kpis": ["FRAUD_RATE"],
                "time_filter": "quarterly_comparison",
                "confidence": 0.90
            }
        },
        {
            "input": "Create a dashboard showing GTV, approval rate, and active users",
            "output": {
                "task_type": "DASHBOARD_CREATION",
                "target_kpis": ["GROSS_TRANSACTION_VALUE", "APPROVAL_RATE", "ACTIVE_USERS"],
                "confidence": 0.92
            }
        }
    ]
    
    # Enhanced SQL examples with comprehensive transaction schema
    SQL_GENERATOR_EXAMPLES = [
        {
            "kpi": "GROSS_TRANSACTION_VALUE",
            "user_query": "Show me total GTV for this month",
            "schema_context": """
            transactions table:
            - amount: DECIMAL(15,2) -- gross transaction amount before fees
            - net_amount: DECIMAL(15,2) -- amount after fees/taxes
            - fee_amount: DECIMAL(15,2) -- platform/processing fees
            - tax_amount: DECIMAL(15,2) -- applicable taxes
            - status: VARCHAR (completed, pending, failed, cancelled)
            - type: VARCHAR (payment, refund, adjustment, chargeback)
            - created_at: TIMESTAMP -- transaction initiation
            - updated_at: TIMESTAMP -- last status update
            """,
            "sql": """
            -- Calculate Gross Transaction Value (GTV) - Sum of all successful gross amounts
            SELECT 
                SUM(amount) as gross_transaction_value,
                SUM(net_amount) as net_transaction_value,
                SUM(fee_amount) as total_fees_collected,
                COUNT(*) as total_transactions,
                AVG(amount) as average_transaction_size
            FROM transactions 
            WHERE status = 'completed' 
                AND type IN ('payment', 'adjustment')
                AND created_at >= DATE_TRUNC('month', CURRENT_DATE)
                AND created_at < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month';
            """,
            "explanation": "Calculates GTV using gross amounts for completed payments and adjustments in current month, excluding refunds and chargebacks"
        },
        {
            "kpi": "NET_REVENUE",
            "user_query": "What's our net revenue after fees and taxes this quarter?",
            "schema_context": """
            transactions table with financial breakdown:
            - amount: gross amount, net_amount: post-fee amount, fee_amount: our fees, tax_amount: taxes
            - status: completed/pending/failed, type: payment/refund/chargeback
            - created_at: transaction timestamp
            """,
            "sql": """
            -- Calculate Net Revenue (Revenue after fees and taxes)
            SELECT 
                SUM(CASE 
                    WHEN type = 'payment' THEN net_amount - tax_amount
                    WHEN type = 'refund' THEN -(net_amount - tax_amount)
                    ELSE 0 
                END) as net_revenue,
                SUM(fee_amount) as total_fees_earned,
                SUM(tax_amount) as total_taxes,
                COUNT(CASE WHEN type = 'payment' AND status = 'completed' THEN 1 END) as successful_payments,
                COUNT(CASE WHEN type = 'refund' AND status = 'completed' THEN 1 END) as refunds_processed
            FROM transactions 
            WHERE status = 'completed'
                AND created_at >= DATE_TRUNC('quarter', CURRENT_DATE)
                AND created_at < DATE_TRUNC('quarter', CURRENT_DATE) + INTERVAL '3 months';
            """,
            "explanation": "Calculates net revenue by summing post-tax net amounts for payments minus refunds, within current quarter"
        },
        {
            "kpi": "FRAUD_RATE",
            "user_query": "Show fraud rate by transaction type and channel",
            "schema_context": """
            transactions table with risk data:
            - is_fraud: BOOLEAN, risk_score: DECIMAL(3,2), status, type, amount
            - channel: VARCHAR (web, mobile, api, pos)
            - category: VARCHAR (retail, subscription, marketplace)
            - created_at: transaction timestamp
            """,
            "sql": """
            -- Calculate Fraud Rate with segmentation by type and channel
            SELECT 
                type as transaction_type,
                channel,
                COUNT(*) as total_transactions,
                COUNT(CASE WHEN is_fraud = true THEN 1 END) as fraud_transactions,
                ROUND(
                    (COUNT(CASE WHEN is_fraud = true THEN 1 END) * 100.0 / COUNT(*)), 
                    2
                ) as fraud_rate_percentage,
                SUM(CASE WHEN is_fraud = true THEN amount ELSE 0 END) as fraud_amount,
                AVG(risk_score) as avg_risk_score
            FROM transactions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY type, channel
            ORDER BY fraud_rate_percentage DESC;
            """,
            "explanation": "Calculates fraud rate segmented by transaction type and channel, including fraud amounts and risk scores"
        },
        {
            "kpi": "APPROVAL_RATE",
            "user_query": "Track approval rates by merchant category over time",
            "schema_context": """
            transactions table with approval data:
            - status: completed/failed/pending, approval_status: approved/declined/manual_review
            - category: merchant category, merchant_id: unique merchant identifier
            - created_at: transaction time, paying_at: actual payment time
            """,
            "sql": """
            -- Calculate Approval Rate by Category with Time Trends
            SELECT 
                category,
                DATE_TRUNC('week', created_at) as week_start,
                COUNT(*) as total_attempts,
                COUNT(CASE WHEN approval_status = 'approved' THEN 1 END) as approved_count,
                ROUND(
                    (COUNT(CASE WHEN approval_status = 'approved' THEN 1 END) * 100.0 / COUNT(*)), 
                    2
                ) as approval_rate_percentage,
                AVG(EXTRACT(EPOCH FROM (paying_at - created_at))/60) as avg_processing_time_minutes
            FROM transactions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '8 weeks'
                AND approval_status IN ('approved', 'declined')
            GROUP BY category, DATE_TRUNC('week', created_at)
            ORDER BY week_start DESC, approval_rate_percentage DESC;
            """,
            "explanation": "Tracks weekly approval rates by merchant category with processing time metrics"
        },
        {
            "kpi": "TRANSACTION_VOLUME",
            "user_query": "Daily transaction volume trends with channel breakdown",
            "schema_context": """
            transactions with volume metrics:
            - amount: transaction value, status: completed/failed
            - channel: web/mobile/api/pos, type: payment/refund
            - created_at: transaction timestamp, updated_at: last modification
            """,
            "sql": """
            -- Daily Transaction Volume Trends by Channel
            SELECT 
                DATE(created_at) as transaction_date,
                channel,
                COUNT(CASE WHEN status = 'completed' AND type = 'payment' THEN 1 END) as successful_payments,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_transactions,
                SUM(CASE WHEN status = 'completed' AND type = 'payment' THEN amount ELSE 0 END) as daily_volume,
                ROUND(
                    (COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*)), 
                    2
                ) as success_rate_percentage,
                COUNT(DISTINCT DATE_TRUNC('hour', created_at)) as active_hours
            FROM transactions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at), channel
            ORDER BY transaction_date DESC, daily_volume DESC;
            """,
            "explanation": "Analyzes daily transaction volumes and success rates by channel with hourly activity patterns"
        },
        {
            "kpi": "ACTIVE_USERS",
            "user_query": "Monthly active users and user engagement metrics",
            "schema_context": """
            transactions with user data:
            - user_id: unique user identifier, amount: transaction value
            - status: completed/failed, type: payment/refund
            - created_at: transaction time, channel: interaction channel
            """,
            "sql": """
            -- Monthly Active Users with Engagement Metrics
            SELECT 
                DATE_TRUNC('month', created_at) as month_period,
                COUNT(DISTINCT user_id) as monthly_active_users,
                COUNT(DISTINCT CASE WHEN status = 'completed' THEN user_id END) as paying_users,
                SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as total_revenue,
                ROUND(
                    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) / 
                    COUNT(DISTINCT CASE WHEN status = 'completed' THEN user_id END), 
                    2
                ) as avg_revenue_per_user,
                COUNT(*) as total_transactions,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT user_id), 2) as avg_transactions_per_user
            FROM transactions 
            WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months')
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY month_period DESC;
            """,
            "explanation": "Calculates monthly active users with revenue and engagement metrics including ARPU"
        },
        {
            "kpi": "PAYMENT_SUCCESS_RATE",
            "user_query": "Payment success rate by time of day and payment method",
            "schema_context": """
            transactions with payment details:
            - status: completed/failed/timeout, payment_method: card/bank/wallet
            - created_at: attempt time, paying_at: successful payment time
            - expired_at: payment window expiry, error_code: failure reason
            """,
            "sql": """
            -- Payment Success Rate Analysis by Time and Method
            SELECT 
                EXTRACT(HOUR FROM created_at) as hour_of_day,
                payment_method,
                COUNT(*) as total_attempts,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_payments,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_payments,
                COUNT(CASE WHEN status = 'timeout' OR created_at > expired_at THEN 1 END) as timeout_payments,
                ROUND(
                    (COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*)), 
                    2
                ) as success_rate_percentage,
                AVG(CASE WHEN paying_at IS NOT NULL THEN 
                    EXTRACT(EPOCH FROM (paying_at - created_at)) ELSE NULL END) as avg_success_time_seconds
            FROM transactions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                AND payment_method IS NOT NULL
            GROUP BY EXTRACT(HOUR FROM created_at), payment_method
            ORDER BY hour_of_day, success_rate_percentage DESC;
            """,
            "explanation": "Analyzes payment success rates by hour and payment method including timing metrics"
        }
    ]
    
    RECOMMENDATION_EXAMPLES = [
        {
            "data_context": {
                "fraud_rate": 2.5,
                "approval_rate": 87.3,
                "gtv": 1250000,
                "trend": "increasing_fraud"
            },
            "recommendation": {
                "descriptive": "Fraud rate has increased to 2.5%, which is above industry average of 1.8%",
                "predictive": "If current trend continues, fraud rate may reach 3.2% next quarter",
                "prescriptive": [
                    "Implement additional ML-based fraud detection rules",
                    "Review and update risk scoring algorithms",
                    "Increase monitoring for high-risk transaction patterns"
                ],
                "priority": "high",
                "impact": "Reducing fraud rate by 1% could save approximately $125,000 monthly"
            }
        }
    ]
    
    @classmethod
    def get_router_prompt(cls, user_input: str) -> str:
        """Generate router prompt with few-shot examples."""
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in cls.ROUTER_EXAMPLES
        ])
        
        return f"""
        You are an intelligent router for a financial analytics system. Analyze user input and determine the task type and target KPIs.

        Examples:
        {examples_text}

        Available task types: KPI_ANALYSIS, TREND_ANALYSIS, COMPARATIVE_ANALYSIS, CUSTOM_QUERY, DASHBOARD_CREATION, RECOMMENDATION_GENERATION

        Available KPIs: {', '.join([kpi.name for kpi in FinancialKPI])}

        User Input: {user_input}

        Return a JSON object with: task_type, target_kpis (list), time_filter (if mentioned), confidence (0-1).
        """
    
    @classmethod
    def get_sql_prompt(cls, kpi: FinancialKPI, schema: Dict, context: Dict) -> str:
        """Generate comprehensive SQL generation prompt with enhanced few-shot examples."""
        
        # Filter examples relevant to the requested KPI
        relevant_examples = [ex for ex in cls.SQL_GENERATOR_EXAMPLES 
                           if kpi.name in ex['kpi'] or any(keyword in kpi.name.lower() 
                           for keyword in ['transaction', 'revenue', 'fraud', 'approval', 'user', 'payment'])]
        
        # If no specific examples, use general transaction examples
        if not relevant_examples:
            relevant_examples = cls.SQL_GENERATOR_EXAMPLES[:3]
        
        examples_text = "\n" + "="*80 + "\n".join([
            f"""
EXAMPLE {i+1}:
KPI: {ex['kpi']}
User Query: {ex.get('user_query', 'Calculate ' + ex['kpi'])}
Schema Context: {ex['schema_context']}

Generated SQL:
{ex['sql']}

Explanation: {ex['explanation']}
{"="*80}"""
            for i, ex in enumerate(relevant_examples)
        ])
        
        kpi_info = kpi.value
        
        # Enhanced schema context with transaction details
        enhanced_schema = cls._build_enhanced_schema_context(schema)
        
        # Time filter parsing
        time_context = cls._parse_time_context(context)
        
        return f"""
You are an expert PostgreSQL database analyst specializing in financial transaction analysis. 
Generate optimized, accurate SQL queries for financial KPIs with deep understanding of transaction data structure.

=== TRANSACTION DATA STRUCTURE UNDERSTANDING ===
Financial transactions typically include:

AMOUNT FIELDS:
- amount/gross_amount: Original transaction value before any deductions
- net_amount: Amount after fees and taxes (amount - fee_amount - tax_amount)  
- fee_amount: Platform/processing fees charged
- tax_amount: Applicable taxes

CLASSIFICATION FIELDS:
- status: completed, pending, failed, cancelled, timeout
- type: payment, refund, adjustment, chargeback, reversal
- category: retail, subscription, marketplace, bill_payment, etc.
- channel: web, mobile, api, pos (point of sale)
- payment_method: card, bank_transfer, digital_wallet, etc.

TEMPORAL FIELDS:
- created_at: When transaction was initiated
- updated_at: Last status/data modification
- paying_at: When payment was actually processed (for completed transactions)
- expired_at: Payment window expiration time

RISK & IDENTITY:
- is_fraud: Boolean fraud flag
- risk_score: Numeric risk assessment (0.0-1.0)
- user_id: Customer identifier
- merchant_id: Merchant/vendor identifier

{examples_text}

=== CURRENT TASK ===
KPI: {kpi.name}
Description: {kpi_info['description']}
Calculation Method: {kpi_info['calculation']}

User Input: {context.get('user_input', 'Calculate ' + kpi.name)}
{time_context}

Database Schema Available:
{enhanced_schema}

=== SQL GENERATION REQUIREMENTS ===
1. **Accuracy**: Use correct column names and table relationships
2. **Performance**: Include appropriate WHERE clauses, indexes, and limits
3. **Completeness**: Handle edge cases (nulls, different statuses, time zones)
4. **Financial Logic**: Understand payment vs refund vs chargeback implications
5. **Time Handling**: Use proper PostgreSQL date functions and time zones
6. **Aggregation**: Provide meaningful breakdowns and totals
7. **Comments**: Include clear explanations of business logic

=== VALIDATION CHECKLIST ===
- ✓ Column names match schema exactly
- ✓ Table names are correct
- ✓ Proper handling of transaction statuses
- ✓ Correct financial calculations (gross vs net)
- ✓ Time filter logic is accurate
- ✓ GROUP BY includes all non-aggregate SELECT fields
- ✓ Performance considerations (WHERE before GROUP BY)

Generate a PostgreSQL query that accurately calculates {kpi.name}.

Return JSON with:
{{
    "sql_query": "-- Complete, optimized PostgreSQL query",
    "explanation": "Detailed explanation of the calculation logic and business rules",
    "expected_columns": ["list", "of", "output", "column", "names"],
    "key_assumptions": ["assumptions made about data and business rules"],
    "performance_notes": "Notes about query optimization and expected performance"
}}
        """
    
    @classmethod
    def _build_enhanced_schema_context(cls, schema: Dict) -> str:
        """Build enhanced schema context with transaction-specific details."""
        if not schema:
            return "Schema information not available - using standard transaction table assumptions"
        
        schema_text = "DATABASE SCHEMA:\n"
        for table_name, table_info in schema.items():
            schema_text += f"\nTable: {table_name}\n"
            if isinstance(table_info, dict) and 'columns' in table_info:
                for col_name, col_info in table_info['columns'].items():
                    data_type = col_info.get('data_type', 'UNKNOWN')
                    nullable = "NULL" if col_info.get('is_nullable', True) else "NOT NULL"
                    schema_text += f"  - {col_name}: {data_type} ({nullable})\n"
                    
                    # Add contextual notes for financial fields
                    if 'amount' in col_name.lower():
                        schema_text += f"    /* Financial amount field - consider currency and precision */\n"
                    elif col_name.lower() in ['status', 'type', 'category']:
                        schema_text += f"    /* Classification field - use exact string matching */\n"
                    elif 'created_at' in col_name.lower() or 'time' in col_name.lower():
                        schema_text += f"    /* Timestamp field - consider timezone handling */\n"
            else:
                schema_text += f"  Columns: {table_info}\n"
        
        return schema_text
    
    @classmethod 
    def _parse_time_context(cls, context: Dict) -> str:
        """Parse and format time context from user input."""
        time_filter = context.get('time_filter')
        user_input = context.get('user_input', '').lower()
        
        time_context = "\nTIME CONTEXT:\n"
        
        if time_filter:
            time_context += f"Detected time filter: {time_filter}\n"
        
        # Detect common time patterns in user input
        time_patterns = {
            'today': "WHERE created_at >= CURRENT_DATE",
            'yesterday': "WHERE created_at >= CURRENT_DATE - INTERVAL '1 day' AND created_at < CURRENT_DATE",
            'this week': "WHERE created_at >= DATE_TRUNC('week', CURRENT_DATE)",
            'last week': "WHERE created_at >= DATE_TRUNC('week', CURRENT_DATE - INTERVAL '7 days') AND created_at < DATE_TRUNC('week', CURRENT_DATE)",
            'this month': "WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)",
            'last month': "WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)",
            'this quarter': "WHERE created_at >= DATE_TRUNC('quarter', CURRENT_DATE)",
            'last quarter': "WHERE created_at >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months') AND created_at < DATE_TRUNC('quarter', CURRENT_DATE)",
            'this year': "WHERE created_at >= DATE_TRUNC('year', CURRENT_DATE)",
            'last 30 days': "WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'"
        }
        
        detected_patterns = []
        for pattern, sql_fragment in time_patterns.items():
            if pattern in user_input:
                detected_patterns.append(f"'{pattern}' -> {sql_fragment}")
        
        if detected_patterns:
            time_context += "Detected time patterns:\n" + "\n".join(detected_patterns)
        else:
            time_context += "No specific time pattern detected - default to last 30 days unless specified otherwise"
        
        return time_context
    
    @classmethod
    def get_recommendation_prompt(cls, analysis_results: Dict, schema_context: Dict) -> str:
        """Generate recommendation prompt with few-shot examples."""
        examples_text = "\n\n".join([
            f"Data Context: {ex['data_context']}\nRecommendation: {ex['recommendation']}"
            for ex in cls.RECOMMENDATION_EXAMPLES
        ])
        
        return f"""
        You are a senior financial strategy consultant. Provide actionable insights using the 3-tier analytics framework.

        Examples of high-quality recommendations:
        {examples_text}

        Current Analysis Results:
        {analysis_results}

        Database Context:
        {schema_context}

        Provide recommendations using this structure:
        1. **Descriptive Analytics**: What happened? (key findings)
        2. **Predictive Analytics**: What will likely happen? (trends and projections)
        3. **Prescriptive Analytics**: What should we do? (specific actions)

        Include priority level, estimated impact, and implementation timeline.
        Return structured JSON with descriptive, predictive, prescriptive, priority, and impact fields.
        """
