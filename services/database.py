"""
Enhanced database services with improved schema analysis and execution.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from core.types import FinancialKPI

logger = logging.getLogger(__name__)


class DatabaseService:
    """Enhanced database service with schema intelligence and query optimization."""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection_pool = None
        self._schema_cache = None
        
    async def get_connection(self):
        """Get database connection with error handling."""
        try:
            return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def get_enhanced_schema(self) -> Dict:
        """Get enhanced schema with business intelligence and relationships."""
        if self._schema_cache:
            return self._schema_cache
            
        conn = await self.get_connection()
        cursor = conn.cursor()
        
        schema_info = {
            'tables': {},
            'relationships': [],
            'business_intelligence': {},
            'performance_hints': {}
        }
        
        try:
            # Get tables and columns
            cursor.execute("""
                SELECT 
                    t.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    tc.constraint_type
                FROM information_schema.tables t
                LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
                LEFT JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name
                WHERE t.table_schema = 'public'
                ORDER BY t.table_name, c.ordinal_position
            """)
            
            results = cursor.fetchall()
            
            # Process results
            for row in results:
                table_name = row['table_name']
                if table_name not in schema_info['tables']:
                    schema_info['tables'][table_name] = {
                        'columns': [],
                        'constraints': [],
                        'business_context': None,
                        'sample_data': None
                    }
                
                if row['column_name']:
                    schema_info['tables'][table_name]['columns'].append({
                        'name': row['column_name'],
                        'type': row['data_type'],
                        'nullable': row['is_nullable'],
                        'default': row['column_default']
                    })
                
                if row['constraint_type']:
                    schema_info['tables'][table_name]['constraints'].append(row['constraint_type'])
            
            # Add business intelligence for each table
            for table_name in schema_info['tables']:
                columns = [col['name'] for col in schema_info['tables'][table_name]['columns']]
                schema_info['tables'][table_name]['business_context'] = await self._analyze_business_context(
                    table_name, columns, cursor
                )
                
                # Get sample data for better context
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_data = cursor.fetchall()
                    schema_info['tables'][table_name]['sample_data'] = [dict(row) for row in sample_data]
                except Exception as e:
                    logger.warning(f"Could not fetch sample data for {table_name}: {e}")
            
            # Detect relationships
            schema_info['relationships'] = await self._detect_relationships(cursor)
            
        finally:
            cursor.close()
            conn.close()
        
        self._schema_cache = schema_info
        return schema_info
    
    async def _analyze_business_context(self, table_name: str, columns: List[str], cursor) -> Dict:
        """Enhanced business context analysis with financial intelligence."""
        table_lower = table_name.lower()
        columns_lower = [c.lower() for c in columns]
        
        # Financial domain keywords
        financial_patterns = {
            'transaction': ['transaction', 'payment', 'transfer', 'charge'],
            'user': ['user', 'customer', 'client', 'account'],
            'merchant': ['merchant', 'vendor', 'seller', 'business'],
            'amount': ['amount', 'value', 'price', 'fee', 'cost'],
            'status': ['status', 'state', 'condition'],
            'time': ['created_at', 'updated_at', 'timestamp', 'date']
        }
        
        # Identify table purpose
        context = {
            'primary_domain': 'general',
            'table_purpose': 'unknown',
            'identified_kpis': [],
            'key_columns': {},
            'data_quality_indicators': {}
        }
        
        # Analyze patterns
        for pattern_type, keywords in financial_patterns.items():
            matches = []
            for keyword in keywords:
                if keyword in table_lower or any(keyword in col for col in columns_lower):
                    matches.extend([col for col in columns if keyword in col.lower()])
            if matches:
                context['key_columns'][pattern_type] = matches
        
        # Determine domain and purpose
        if any(kw in table_lower for kw in financial_patterns['transaction']):
            context['primary_domain'] = 'finance'
            context['table_purpose'] = 'transaction_log'
        elif any(kw in table_lower for kw in financial_patterns['user']):
            context['primary_domain'] = 'customer'
            context['table_purpose'] = 'user_dimension'
        elif any(kw in table_lower for kw in financial_patterns['merchant']):
            context['primary_domain'] = 'business'
            context['table_purpose'] = 'merchant_dimension'
        
        # Map to KPIs
        for kpi in FinancialKPI:
            kpi_name = kpi.name.lower()
            if any(word in kpi_name for word in table_lower.split('_')) or \
               any(any(word in col for col in columns_lower) for word in kpi_name.split('_')):
                context['identified_kpis'].append(kpi.name)
        
        # Get data quality indicators
        try:
            cursor.execute(f"SELECT COUNT(*) as row_count FROM {table_name}")
            result = cursor.fetchone()
            context['data_quality_indicators']['row_count'] = result['row_count']
            
            # Check for null values in key columns
            if 'amount' in context['key_columns']:
                amount_col = context['key_columns']['amount'][0]
                cursor.execute(f"SELECT COUNT(*) as null_count FROM {table_name} WHERE {amount_col} IS NULL")
                result = cursor.fetchone()
                context['data_quality_indicators']['null_amounts'] = result['null_count']
                
        except Exception as e:
            logger.warning(f"Could not get data quality indicators for {table_name}: {e}")
        
        return context
    
    async def _detect_relationships(self, cursor) -> List[Dict]:
        """Detect foreign key relationships between tables."""
        try:
            cursor.execute("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
            """)
            
            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    'from_table': row['table_name'],
                    'from_column': row['column_name'],
                    'to_table': row['foreign_table_name'],
                    'to_column': row['foreign_column_name']
                })
            
            return relationships
        except Exception as e:
            logger.warning(f"Could not detect relationships: {e}")
            return []
    
    async def execute_query(self, sql_query: str) -> Any:
        """Execute SQL query with enhanced error handling and optimization."""
        conn = None
        cursor = None
        
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Log query for debugging
            logger.info(f"Executing query: {sql_query}")
            
            cursor.execute(sql_query)
            result = cursor.fetchall()
            
            # Format result
            if not result:
                return None
            elif len(result) == 1 and len(result[0]) == 1:
                return list(result[0].values())[0]
            else:
                return [dict(row) for row in result]
                
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error executing query: {e}")
            return {"error": f"Database error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            return {"error": f"Execution error: {str(e)}"}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    async def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query without executing it."""
        conn = None
        cursor = None
        
        try:
            conn = await self.get_connection()
            cursor = conn.cursor()
            
            # Use EXPLAIN to validate without executing
            explain_query = f"EXPLAIN {sql_query}"
            cursor.execute(explain_query)
            plan = cursor.fetchall()
            
            return {
                "valid": True,
                "execution_plan": [dict(row) for row in plan],
                "estimated_cost": self._extract_cost_from_plan(plan)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def _extract_cost_from_plan(self, plan: List) -> Optional[float]:
        """Extract estimated cost from execution plan."""
        try:
            if plan and len(plan) > 0:
                plan_text = plan[0].get('QUERY PLAN', '')
                if 'cost=' in plan_text:
                    cost_part = plan_text.split('cost=')[1].split(')')[0]
                    return float(cost_part.split('..')[1])
        except Exception:
            pass
        return None
