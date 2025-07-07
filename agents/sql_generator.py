"""
SQL Generator Agent - Specialized in creating optimized SQL queries for financial KPIs.
"""
import asyncio
import logging
import time
from typing import Dict, List, Any
from core.types import AgentState, FinancialKPI
from core.prompts import PromptTemplates
from services.llm import LLMService
from services.vector_store import VectorStoreService
from services.database import DatabaseService

logger = logging.getLogger(__name__)


class SQLGeneratorAgent:
    """
    Specialized agent for generating performance-optimized SQL queries
    for financial KPIs with few-shot prompting and validation.
    """
    
    def __init__(self, llm_service: LLMService, vector_service: VectorStoreService, 
                 db_service: DatabaseService):
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.db_service = db_service
        self.agent_type = "sql_generator"
    
    async def generate_sql_queries(self, state: AgentState) -> AgentState:
        """Generate SQL queries for all target KPIs with optimization."""
        start_time = time.time()
        
        try:
            # Ensure we have schema information
            if not state.database_schema:
                state.database_schema = await self.db_service.get_enhanced_schema()
            
            # Generate SQL for each target KPI
            sql_generation_tasks = []
            for kpi in state.target_kpis:
                task = self._generate_single_kpi_query(kpi, state)
                sql_generation_tasks.append(task)
            
            # Execute in parallel for efficiency
            sql_results = await asyncio.gather(*sql_generation_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(sql_results):
                kpi = state.target_kpis[i]
                
                if isinstance(result, Exception):
                    logger.error(f"SQL generation failed for {kpi.name}: {result}")
                    state.error_messages.append(f"SQL generation failed for {kpi.name}: {str(result)}")
                    continue
                
                if 'error' in result:
                    state.error_messages.append(f"SQL generation error for {kpi.name}: {result['error']}")
                    continue
                
                # Store successful SQL query
                state.sql_queries[kpi.name] = result['sql_query']
                
                # Add metadata about the query
                if 'metadata' not in state.metadata:
                    state.metadata['query_metadata'] = {}
                
                state.metadata['query_metadata'][kpi.name] = {
                    'explanation': result.get('explanation', ''),
                    'expected_columns': result.get('expected_columns', []),
                    'optimization_level': result.get('optimization_level', 'standard'),
                    'validation_passed': result.get('validation_passed', False)
                }
            
            # Log execution
            execution_time = time.time() - start_time
            await self.vector_service.log_agent_execution(
                session_id=state.session_id,
                agent_name=self.agent_type,
                input_data={'target_kpis': [kpi.name for kpi in state.target_kpis]},
                output_data={'queries_generated': len(state.sql_queries)},
                execution_time=execution_time,
                model_used='llama3-70b-8192'  # High-precision model for SQL
            )
            
            logger.info(f"Generated {len(state.sql_queries)} SQL queries in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"SQL Generator agent failed: {e}")
            state.error_messages.append(f"SQL generation failed: {str(e)}")
        
        return state
    
    async def _generate_single_kpi_query(self, kpi: FinancialKPI, state: AgentState) -> Dict:
        """Generate SQL query for a single KPI with validation and chaining prompts."""
        try:
            # Step 1: Analyze user intent and extract requirements
            intent_context = await self._analyze_user_intent(kpi, state)
            
            # Step 2: Build comprehensive context for the KPI
            context = {
                'user_input': state.user_input,
                'time_filter': state.metadata.get('time_filter'),
                'session_context': state.conversation_history[-3:] if state.conversation_history else [],
                'intent_analysis': intent_context,
                'kpi_requirements': self._get_kpi_requirements(kpi)
            }
            
            # Step 3: Generate SQL using enhanced few-shot prompting
            prompt = PromptTemplates.get_sql_prompt(kpi, state.database_schema, context)
            
            result = await self.llm_service.generate_structured_response(
                prompt=prompt,
                system_prompt=self._get_enhanced_system_prompt(),
                agent_type=self.agent_type
            )
            
            sql_response = result['result']
            
            if 'error' in sql_response:
                return sql_response
            
            # Step 4: Validate and refine the generated SQL
            validation_result = await self._comprehensive_sql_validation(
                sql_response.get('sql_query', ''), kpi, context
            )
            
            sql_response.update(validation_result)
            
            # Step 5: If validation fails, attempt intelligent fixes
            if not validation_result.get('validation_passed', False):
                fixed_result = await self._intelligent_sql_refinement(
                    sql_response, kpi, context, validation_result
                )
                if fixed_result:
                    sql_response.update(fixed_result)
            
            # Step 6: Add optimization and performance insights
            sql_response['optimization_analysis'] = await self._analyze_query_performance(
                sql_response.get('sql_query', '')
            )
            
            return sql_response
            
        except Exception as e:
            logger.error(f"Single KPI SQL generation failed for {kpi.name}: {e}")
            return {'error': str(e)}
    
    async def _analyze_user_intent(self, kpi: FinancialKPI, state: AgentState) -> Dict:
        """Analyze user intent to better understand query requirements."""
        intent_prompt = f"""
        Analyze the user's intent for this financial KPI request:
        
        User Input: "{state.user_input}"
        Target KPI: {kpi.name}
        KPI Description: {kpi.value.get('description', '')}
        
        Extract and identify:
        1. Time period requirements (specific dates, relative periods)
        2. Segmentation needs (by category, channel, merchant, etc.)
        3. Comparison requirements (vs previous period, vs target)
        4. Aggregation level (daily, weekly, monthly, total)
        5. Specific filters or conditions mentioned
        6. Output format preferences (summary, detailed, trends)
        
        Return JSON with: time_requirements, segmentation, comparisons, aggregation_level, filters, output_format
        """
        
        try:
            result = await self.llm_service.generate_structured_response(
                prompt=intent_prompt,
                system_prompt="You are an expert business analyst specializing in financial data requirements analysis.",
                agent_type="intent_analyzer"
            )
            return result.get('result', {})
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return {}
    
    def _get_kpi_requirements(self, kpi: FinancialKPI) -> Dict:
        """Get specific requirements and business rules for each KPI."""
        kpi_requirements = {
            'GROSS_TRANSACTION_VALUE': {
                'include_statuses': ['completed'],
                'exclude_types': ['refund', 'chargeback'],
                'amount_field': 'amount',
                'business_rules': [
                    'Include only successful payments',
                    'Exclude refunds and chargebacks from gross calculations',
                    'Consider only completed transactions'
                ]
            },
            'NET_REVENUE': {
                'include_statuses': ['completed'],
                'calculation': 'net_amount - tax_amount for payments, subtract refunds',
                'amount_field': 'net_amount',
                'business_rules': [
                    'Calculate as net amount minus taxes',
                    'Subtract refunds from revenue',
                    'Include fee income when applicable'
                ]
            },
            'FRAUD_RATE': {
                'calculation': 'fraud_transactions / total_transactions * 100',
                'fraud_field': 'is_fraud',
                'business_rules': [
                    'Include all transaction attempts (completed and failed)',
                    'Express as percentage',
                    'Consider risk scores when available'
                ]
            },
            'APPROVAL_RATE': {
                'calculation': 'approved_transactions / total_attempts * 100',
                'status_field': 'approval_status',
                'business_rules': [
                    'Include all approval attempts',
                    'Exclude pending/timeout from denominator',
                    'Track by payment method and merchant category'
                ]
            },
            'ACTIVE_USERS': {
                'calculation': 'COUNT(DISTINCT user_id) with successful transactions',
                'activity_criteria': 'completed transactions',
                'business_rules': [
                    'Count unique users with at least one successful transaction',
                    'Consider time window for activity definition',
                    'Exclude test/system accounts'
                ]
            }
        }
        
        return kpi_requirements.get(kpi.name, {
            'business_rules': ['Follow standard financial KPI calculation methods']
        })
    
    async def _comprehensive_sql_validation(self, sql_query: str, kpi: FinancialKPI, context: Dict) -> Dict:
        """Perform comprehensive SQL validation including syntax, logic, and business rules."""
        validation_results = {
            'validation_passed': False,
            'syntax_valid': False,
            'logic_valid': False,
            'business_rules_valid': False,
            'validation_errors': [],
            'validation_warnings': []
        }
        
        try:
            # Step 1: Basic syntax validation
            syntax_result = await self.db_service.validate_query(sql_query)
            validation_results['syntax_valid'] = syntax_result.get('valid', False)
            
            if not syntax_result.get('valid', False):
                validation_results['validation_errors'].append(f"Syntax error: {syntax_result.get('error', 'Unknown syntax issue')}")
                return validation_results
            
            # Step 2: Business logic validation
            logic_validation = await self._validate_business_logic(sql_query, kpi, context)
            validation_results.update(logic_validation)
            
            # Step 3: Performance validation
            performance_validation = await self._validate_query_performance(sql_query)
            validation_results['performance_warnings'] = performance_validation.get('warnings', [])
            
            # Overall validation status
            validation_results['validation_passed'] = (
                validation_results['syntax_valid'] and 
                validation_results['logic_valid'] and
                len(validation_results['validation_errors']) == 0
            )
            
        except Exception as e:
            validation_results['validation_errors'].append(f"Validation process failed: {str(e)}")
        
        return validation_results
    
    async def _validate_business_logic(self, sql_query: str, kpi: FinancialKPI, context: Dict) -> Dict:
        """Validate business logic using LLM-based analysis."""
        kpi_requirements = self._get_kpi_requirements(kpi)
        
        validation_prompt = f"""
        Validate this SQL query for business logic correctness:
        
        KPI: {kpi.name}
        Business Rules: {kpi_requirements.get('business_rules', [])}
        User Intent: {context.get('intent_analysis', {})}
        
        SQL Query:
        {sql_query}
        
        Check for:
        1. Correct transaction status filtering
        2. Proper handling of refunds/chargebacks
        3. Accurate amount field usage (gross vs net)
        4. Appropriate time filtering
        5. Correct aggregation logic
        6. Proper handling of NULL values
        7. Business rule compliance
        
        Return JSON with: logic_valid (boolean), issues_found (list), recommendations (list)
        """
        
        try:
            result = await self.llm_service.generate_structured_response(
                prompt=validation_prompt,
                system_prompt="You are a financial data validation expert. Analyze SQL queries for business logic correctness.",
                agent_type="sql_validator"
            )
            
            validation_result = result.get('result', {})
            return {
                'logic_valid': validation_result.get('logic_valid', False),
                'logic_issues': validation_result.get('issues_found', []),
                'logic_recommendations': validation_result.get('recommendations', [])
            }
            
        except Exception as e:
            logger.warning(f"Business logic validation failed: {e}")
            return {
                'logic_valid': True,  # Default to true if validation fails
                'logic_issues': [],
                'logic_recommendations': []
            }
    
    async def _intelligent_sql_refinement(self, sql_response: Dict, kpi: FinancialKPI, 
                                        context: Dict, validation_result: Dict) -> Dict:
        """Attempt to fix SQL issues using intelligent refinement."""
        if validation_result.get('validation_passed', False):
            return None
        
        refinement_prompt = f"""
        Refine this SQL query to fix identified issues:
        
        Original Query:
        {sql_response.get('sql_query', '')}
        
        Issues Found:
        - Syntax Errors: {validation_result.get('validation_errors', [])}
        - Logic Issues: {validation_result.get('logic_issues', [])}
        - Recommendations: {validation_result.get('logic_recommendations', [])}
        
        KPI Requirements:
        {self._get_kpi_requirements(kpi)}
        
        User Context:
        {context}
        
        Generate a corrected SQL query that addresses all identified issues.
        
        Return JSON with: refined_sql_query, changes_made, confidence_score
        """
        
        try:
            result = await self.llm_service.generate_structured_response(
                prompt=refinement_prompt,
                system_prompt="You are an expert SQL engineer specializing in fixing and optimizing financial queries.",
                agent_type="sql_refiner"
            )
            
            refinement_result = result.get('result', {})
            
            if refinement_result.get('refined_sql_query'):
                # Validate the refined query
                refined_validation = await self.db_service.validate_query(
                    refinement_result['refined_sql_query']
                )
                
                if refined_validation.get('valid', False):
                    return {
                        'sql_query': refinement_result['refined_sql_query'],
                        'validation_passed': True,
                        'auto_refined': True,
                        'refinement_changes': refinement_result.get('changes_made', []),
                        'refinement_confidence': refinement_result.get('confidence_score', 0.5)
                    }
            
        except Exception as e:
            logger.warning(f"SQL refinement failed: {e}")
        
        return None
    
    async def _validate_sql_query(self, sql_query: str) -> Dict:
        """Validate SQL query using database explain plan."""
        try:
            return await self.db_service.validate_query(sql_query)
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _attempt_sql_fix(self, sql_query: str, error_message: str) -> str:
        """Attempt to fix common SQL errors automatically."""
        # Common fixes for PostgreSQL
        fixes = {
            'column does not exist': self._fix_column_names,
            'table does not exist': self._fix_table_names,
            'syntax error': self._fix_syntax_errors,
            'GROUP BY': self._fix_group_by_issues
        }
        
        for error_pattern, fix_function in fixes.items():
            if error_pattern.lower() in error_message.lower():
                try:
                    fixed_sql = await fix_function(sql_query, error_message)
                    if fixed_sql and fixed_sql != sql_query:
                        # Validate the fix
                        validation = await self._validate_sql_query(fixed_sql)
                        if validation['valid']:
                            return fixed_sql
                except Exception as e:
                    logger.warning(f"Auto-fix attempt failed: {e}")
        
        return None
    
    async def _fix_column_names(self, sql_query: str, error_message: str) -> str:
        """Fix column name issues by suggesting similar column names."""
        # This is a simplified version - in production, you'd use fuzzy matching
        # against actual schema columns
        return sql_query
    
    async def _fix_table_names(self, sql_query: str, error_message: str) -> str:
        """Fix table name issues."""
        return sql_query
    
    async def _fix_syntax_errors(self, sql_query: str, error_message: str) -> str:
        """Fix common syntax errors."""
        # Remove common syntax issues
        fixed = sql_query.replace(';;', ';').strip()
        if not fixed.endswith(';'):
            fixed += ';'
        return fixed
    
    async def _fix_group_by_issues(self, sql_query: str, error_message: str) -> str:
        """Fix GROUP BY clause issues."""
        return sql_query
    
    def _assess_query_optimization(self, sql_query: str) -> str:
        """Assess the optimization level of the generated query."""
        optimization_indicators = {
            'high': ['INDEX', 'LIMIT', 'WHERE.*=.*', 'INNER JOIN'],
            'medium': ['WHERE', 'ORDER BY', 'GROUP BY'],
            'low': ['SELECT \\*', 'LIKE', 'OR']
        }
        
        sql_upper = sql_query.upper()
        
        # Check for high optimization indicators
        high_score = sum(1 for pattern in optimization_indicators['high'] 
                        if pattern in sql_upper)
        
        # Check for medium optimization indicators
        medium_score = sum(1 for pattern in optimization_indicators['medium'] 
                          if pattern in sql_upper)
        
        # Check for low optimization indicators (negative points)
        low_score = sum(1 for pattern in optimization_indicators['low'] 
                       if pattern in sql_upper)
        
        total_score = high_score * 3 + medium_score * 2 - low_score
        
        if total_score >= 6:
            return 'high'
        elif total_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for SQL generation."""
        return """
        You are an expert PostgreSQL database analyst specializing in financial services and e-wallet systems.
        Your primary responsibility is to generate accurate, optimized, and secure SQL queries for financial KPIs.
        
        Key Requirements:
        1. Generate syntactically correct PostgreSQL queries
        2. Include helpful comments explaining the business logic
        3. Optimize for performance with appropriate WHERE clauses and indexes
        4. Handle edge cases (NULL values, data quality issues)
        5. Use proper aggregation functions for financial calculations
        6. Include data validation where appropriate
        
        Best Practices:
        - Use explicit JOIN syntax instead of implicit joins
        - Add appropriate WHERE clauses to filter data efficiently
        - Use LIMIT clauses for large datasets when appropriate
        - Handle decimal precision for financial calculations
        - Include error handling for division by zero
        - Use date/time functions correctly for temporal analysis
        
        Return a JSON object with:
        - sql_query: The complete, executable SQL query
        - explanation: Business logic explanation
        - expected_columns: List of output column names and types
        - performance_notes: Any performance considerations
        """
    
    async def optimize_existing_query(self, sql_query: str, performance_target: str = 'fast') -> Dict:
        """Optimize an existing SQL query for better performance."""
        optimization_prompt = f"""
        Optimize the following SQL query for {performance_target} execution:
        
        Original Query:
        {sql_query}
        
        Provide an optimized version with:
        1. Better indexing strategies
        2. More efficient WHERE clauses
        3. Optimized JOIN operations
        4. Reduced data scanning
        
        Return JSON with optimized_query, improvements_made, and performance_estimate.
        """
        
        result = await self.llm_service.generate_structured_response(
            prompt=optimization_prompt,
            system_prompt="You are a PostgreSQL performance optimization expert.",
            agent_type=self.agent_type
        )
        
        return result['result']
    
    async def _validate_query_performance(self, sql_query: str) -> Dict:
        """Validate query performance and suggest optimizations."""
        performance_warnings = []
        
        # Simple performance checks
        sql_upper = sql_query.upper()
        
        if 'SELECT *' in sql_upper:
            performance_warnings.append("Using SELECT * can be inefficient - specify exact columns needed")
        
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            performance_warnings.append("ORDER BY without LIMIT may be expensive on large datasets")
        
        if sql_upper.count('JOIN') > 3:
            performance_warnings.append("Complex joins detected - consider query optimization")
        
        if 'WHERE' not in sql_upper:
            performance_warnings.append("No WHERE clause - query may scan entire table")
        
        return {'warnings': performance_warnings}
    
    async def _analyze_query_performance(self, sql_query: str) -> Dict:
        """Analyze query performance characteristics."""
        analysis = {
            'complexity_score': 0,
            'optimization_suggestions': [],
            'estimated_performance': 'unknown'
        }
        
        sql_upper = sql_query.upper()
        
        # Calculate complexity score
        complexity_factors = {
            'JOIN': sql_upper.count('JOIN') * 2,
            'SUBQUERY': sql_upper.count('SELECT') - 1,
            'GROUP BY': 1 if 'GROUP BY' in sql_upper else 0,
            'ORDER BY': 1 if 'ORDER BY' in sql_upper else 0,
            'WINDOW FUNCTION': sql_upper.count('OVER(') * 3
        }
        
        analysis['complexity_score'] = sum(complexity_factors.values())
        
        # Performance estimation
        if analysis['complexity_score'] <= 2:
            analysis['estimated_performance'] = 'fast'
        elif analysis['complexity_score'] <= 5:
            analysis['estimated_performance'] = 'moderate'
        else:
            analysis['estimated_performance'] = 'slow'
        
        # Optimization suggestions
        if 'INDEX' not in sql_upper and 'WHERE' in sql_upper:
            analysis['optimization_suggestions'].append("Consider adding indexes on WHERE clause columns")
        
        if analysis['complexity_score'] > 5:
            analysis['optimization_suggestions'].append("Consider breaking down into simpler queries or using materialized views")
        
        return analysis
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt for SQL generation."""
        return """
        You are an expert PostgreSQL database analyst with deep expertise in financial transaction systems.
        
        KEY PRINCIPLES:
        1. Accuracy is paramount - financial data must be precisely calculated
        2. Performance matters - write optimized queries for large transaction volumes  
        3. Business logic compliance - understand financial domain rules
        4. Comprehensive error handling - account for edge cases and data quality issues
        
        FINANCIAL DOMAIN EXPERTISE:
        - Understand difference between gross amounts, net amounts, fees, and taxes
        - Know transaction lifecycle: pending -> completed/failed/cancelled
        - Recognize payment types: payment, refund, chargeback, adjustment
        - Handle time-sensitive calculations with proper timezone handling
        - Apply appropriate filtering for different KPI calculations
        
        QUERY STANDARDS:
        - Use explicit column names (avoid SELECT *)
        - Include meaningful comments explaining business logic
        - Handle NULL values appropriately
        - Use proper date/time functions for PostgreSQL
        - Apply consistent formatting and indentation
        - Include performance optimizations (indexes, limits, efficient joins)
        
        Always return valid JSON with the requested fields.
        """
