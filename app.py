import os
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Database and Vector Store
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM Integration
import groq
from groq import Groq

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    DATABASE_INSIGHT = "database_insight"
    SQL_GENERATOR = "sql_generator"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    ORCHESTRATOR = "orchestrator"

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    database_config: Dict
    conversation_history: List[Dict]
    insights_generated: List[Dict]
    current_context: Dict
    timestamp: datetime

@dataclass
class InsightResult:
    insight_id: str
    insight_type: str
    content: Dict
    metadata: Dict
    confidence_score: float
    timestamp: datetime

class DatabaseSchemaExtractor:
    """Enhanced database schema extraction with deep learning capabilities"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.schema_cache = {}
        
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                cursor_factory=RealDictCursor
            )
            logger.info(f"Connected to database: {self.db_config['database']}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def extract_complete_schema(self) -> Dict:
        """Extract complete database schema with relationships"""
        if not self.connection:
            await self.connect()
        
        cursor = self.connection.cursor()
        
        schema_info = {
            'tables': {},
            'relationships': [],
            'indexes': {},
            'constraints': {},
            'functions': [],
            'views': [],
            'materialized_views': [],
            'metadata': {}
        }
        
        # Extract all tables
        cursor.execute("""
            SELECT table_name, table_type, table_schema
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table['table_name']
            table_info = await self._extract_table_details(table_name)
            schema_info['tables'][table_name] = table_info
        
        # Extract relationships
        schema_info['relationships'] = await self._extract_relationships()
        
        # Extract indexes
        schema_info['indexes'] = await self._extract_indexes()
        
        # Extract constraints
        schema_info['constraints'] = await self._extract_constraints()
        
        # Extract functions and procedures
        schema_info['functions'] = await self._extract_functions()
        
        # Extract views
        schema_info['views'] = await self._extract_views()
        
        # Generate semantic understanding
        schema_info['semantic_analysis'] = await self._generate_semantic_analysis(schema_info)
        
        cursor.close()
        return schema_info
    
    async def _extract_table_details(self, table_name: str) -> Dict:
        """Extract detailed information about a table"""
        cursor = self.connection.cursor()
        
        # Column information
        cursor.execute("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.ordinal_position,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN uk.column_name IS NOT NULL THEN true ELSE false END as is_unique
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_name = %s
            ) pk ON c.column_name = pk.column_name
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'UNIQUE'
                    AND tc.table_name = %s
            ) uk ON c.column_name = uk.column_name
            WHERE c.table_name = %s
            ORDER BY c.ordinal_position
        """, (table_name, table_name, table_name))
        
        columns = cursor.fetchall()
        
        # Get row count and sample data
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()['count']
        
        # Get sample data for analysis
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 10')
        sample_data = cursor.fetchall()
        
        # Analyze data patterns
        data_patterns = await self._analyze_data_patterns(table_name, columns)
        
        table_info = {
            'name': table_name,
            'columns': [dict(col) for col in columns],
            'row_count': row_count,
            'sample_data': [dict(row) for row in sample_data],
            'data_patterns': data_patterns,
            'business_context': await self._infer_business_context(table_name, columns)
        }
        
        cursor.close()
        return table_info
    
    async def _analyze_data_patterns(self, table_name: str, columns: List[Dict]) -> Dict:
        """Analyze data patterns in the table"""
        cursor = self.connection.cursor()
        patterns = {}
        
        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            
            # Basic statistics
            if data_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
                cursor.execute(f"""
                    SELECT 
                        MIN("{col_name}") as min_val,
                        MAX("{col_name}") as max_val,
                        AVG("{col_name}") as avg_val,
                        STDDEV("{col_name}") as std_val,
                        COUNT(DISTINCT "{col_name}") as unique_count
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'numeric',
                        'statistics': dict(result),
                        'distribution': 'normal'  # Could be enhanced with actual distribution analysis
                    }
            
            elif data_type in ['character varying', 'text', 'character']:
                cursor.execute(f"""
                    SELECT 
                        COUNT(DISTINCT "{col_name}") as unique_count,
                        AVG(LENGTH("{col_name}")) as avg_length,
                        MAX(LENGTH("{col_name}")) as max_length
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'categorical',
                        'statistics': dict(result),
                        'cardinality': 'high' if result['unique_count'] > 100 else 'low'
                    }
            
            elif data_type in ['timestamp', 'date', 'time']:
                cursor.execute(f"""
                    SELECT 
                        MIN("{col_name}") as min_date,
                        MAX("{col_name}") as max_date,
                        COUNT(DISTINCT DATE("{col_name}")) as unique_dates
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'temporal',
                        'statistics': dict(result),
                        'time_series': True
                    }
        
        cursor.close()
        return patterns
    
    async def _infer_business_context(self, table_name: str, columns: List[Dict]) -> Dict:
        """Infer business context from table and column names"""
        # Business domain inference based on naming patterns
        business_domains = {
            'financial': ['transaction', 'payment', 'account', 'balance', 'amount', 'price', 'cost'],
            'customer': ['customer', 'client', 'user', 'person', 'contact', 'profile'],
            'product': ['product', 'item', 'inventory', 'stock', 'catalog'],
            'sales': ['sale', 'order', 'purchase', 'revenue', 'profit'],
            'marketing': ['campaign', 'advertisement', 'promotion', 'lead'],
            'operations': ['operation', 'process', 'workflow', 'status'],
            'analytics': ['metric', 'measure', 'kpi', 'score', 'rating']
        }
        
        table_lower = table_name.lower()
        column_names = [col['column_name'].lower() for col in columns]
        
        domain_scores = {}
        for domain, keywords in business_domains.items():
            score = 0
            for keyword in keywords:
                if keyword in table_lower:
                    score += 2
                for col_name in column_names:
                    if keyword in col_name:
                        score += 1
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        return {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'table_purpose': await self._infer_table_purpose(table_name, columns),
            'key_metrics': await self._identify_key_metrics(columns)
        }
    
    async def _infer_table_purpose(self, table_name: str, columns: List[Dict]) -> str:
        """Infer the purpose of the table"""
        column_names = [col['column_name'].lower() for col in columns]
        
        if any('transaction' in name for name in column_names + [table_name.lower()]):
            return 'transaction_log'
        elif any('customer' in name for name in column_names + [table_name.lower()]):
            return 'customer_data'
        elif any('product' in name for name in column_names + [table_name.lower()]):
            return 'product_catalog'
        elif any('order' in name for name in column_names + [table_name.lower()]):
            return 'order_management'
        else:
            return 'data_storage'
    
    async def _identify_key_metrics(self, columns: List[Dict]) -> List[str]:
        """Identify key metrics columns"""
        metric_indicators = ['amount', 'total', 'count', 'sum', 'average', 'rate', 'score', 'value']
        key_metrics = []
        
        for col in columns:
            col_name = col['column_name'].lower()
            if any(indicator in col_name for indicator in metric_indicators):
                key_metrics.append(col['column_name'])
        
        return key_metrics
    
    async def _extract_relationships(self) -> List[Dict]:
        """Extract foreign key relationships"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                tc.constraint_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
        """)
        
        relationships = cursor.fetchall()
        cursor.close()
        return [dict(rel) for rel in relationships]
    
    async def _extract_indexes(self) -> Dict:
        """Extract database indexes"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        
        indexes = cursor.fetchall()
        cursor.close()
        return {idx['indexname']: dict(idx) for idx in indexes}
    
    async def _extract_constraints(self) -> Dict:
        """Extract database constraints"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                tc.table_name,
                tc.constraint_name,
                tc.constraint_type,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_schema = 'public'
        """)
        
        constraints = cursor.fetchall()
        cursor.close()
        return {const['constraint_name']: dict(const) for const in constraints}
    
    async def _extract_functions(self) -> List[Dict]:
        """Extract database functions and procedures"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                routine_name,
                routine_type,
                data_type,
                routine_definition
            FROM information_schema.routines
            WHERE routine_schema = 'public'
        """)
        
        functions = cursor.fetchall()
        cursor.close()
        return [dict(func) for func in functions]
    
    async def _extract_views(self) -> List[Dict]:
        """Extract database views"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                table_name,
                view_definition
            FROM information_schema.views
            WHERE table_schema = 'public'
        """)
        
        views = cursor.fetchall()
        cursor.close()
        return [dict(view) for view in views]
    
    async def _generate_semantic_analysis(self, schema_info: Dict) -> Dict:
        """Generate semantic analysis of the database schema"""
        tables = schema_info['tables']
        relationships = schema_info['relationships']
        
        # Analyze table relationships
        table_graph = {}
        for table_name in tables.keys():
            table_graph[table_name] = {
                'incoming': [],
                'outgoing': [],
                'centrality': 0
            }
        
        for rel in relationships:
            source_table = rel['table_name']
            target_table = rel['foreign_table_name']
            
            if source_table in table_graph:
                table_graph[source_table]['outgoing'].append(target_table)
            if target_table in table_graph:
                table_graph[target_table]['incoming'].append(source_table)
        
        # Calculate centrality scores
        for table_name, graph_info in table_graph.items():
            centrality = len(graph_info['incoming']) + len(graph_info['outgoing'])
            table_graph[table_name]['centrality'] = centrality
        
        # Identify core tables
        core_tables = sorted(table_graph.keys(), 
                           key=lambda x: table_graph[x]['centrality'], 
                           reverse=True)[:3]
        
        return {
            'table_graph': table_graph,
            'core_tables': core_tables,
            'database_complexity': len(tables),
            'relationship_density': len(relationships) / len(tables) if tables else 0
        }

class VectorStoreManager:
    """Manages vector storage for RAG system"""
    
    def __init__(self, mongodb_uri: str, collection_name: str = "bisee_vectors"):
        self.mongodb_uri = mongodb_uri
        self.collection_name = collection_name
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.bisee_db
        self.collection = self.db[collection_name]
        self.sessions_collection = self.db.sessions
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for fast similarity search
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.create_collection(
            name="bisee_insights",
            get_or_create=True
        )
    
    async def store_insight(self, insight: InsightResult, session_id: str):
        """Store insight in both MongoDB and ChromaDB"""
        # Prepare content for embedding
        content_text = json.dumps(insight.content)
        embedding = self.encoder.encode(content_text).tolist()
        
        # Store in MongoDB
        document = {
            'insight_id': insight.insight_id,
            'session_id': session_id,
            'insight_type': insight.insight_type,
            'content': insight.content,
            'metadata': insight.metadata,
            'confidence_score': insight.confidence_score,
            'timestamp': insight.timestamp,
            'embedding': embedding
        }
        
        await asyncio.to_thread(self.collection.insert_one, document)
        
        # Store in ChromaDB for fast retrieval
        self.chroma_collection.add(
            embeddings=[embedding],
            documents=[content_text],
            metadatas=[{
                'insight_id': insight.insight_id,
                'session_id': session_id,
                'insight_type': insight.insight_type,
                'confidence_score': insight.confidence_score
            }],
            ids=[insight.insight_id]
        )
    
    async def retrieve_similar_insights(self, query: str, session_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve similar insights based on query"""
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"session_id": session_id}
        )
        
        # Get full documents from MongoDB
        insight_ids = results['ids'][0] if results['ids'] else []
        insights = []
        
        for insight_id in insight_ids:
            doc = await asyncio.to_thread(
                self.collection.find_one,
                {'insight_id': insight_id}
            )
            if doc:
                insights.append(doc)
        
        return insights
    
    async def get_session_insights(self, session_id: str) -> List[Dict]:
        """Get all insights for a session"""
        cursor = self.collection.find({'session_id': session_id})
        insights = await asyncio.to_thread(list, cursor)
        return insights
    
    async def store_session_context(self, session_context: SessionContext):
        """Store session context in MongoDB"""
        document = {
            'session_id': session_context.session_id,
            'user_id': session_context.user_id,
            'database_config': session_context.database_config,
            'conversation_history': session_context.conversation_history,
            'insights_generated': session_context.insights_generated,
            'current_context': session_context.current_context,
            'timestamp': session_context.timestamp
        }
        
        await asyncio.to_thread(
            self.sessions_collection.replace_one,
            {'session_id': session_context.session_id},
            document,
            upsert=True
        )
    
    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve session context"""
        document = await asyncio.to_thread(
            self.sessions_collection.find_one,
            {'session_id': session_id}
        )
        
        if document:
            return SessionContext(
                session_id=document['session_id'],
                user_id=document['user_id'],
                database_config=document['database_config'],
                conversation_history=document['conversation_history'],
                insights_generated=document['insights_generated'],
                current_context=document['current_context'],
                timestamp=document['timestamp']
            )
        return None

class LLMManager:
    """Manages Groq LLM interactions"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Use the larger model for better insights
    
    async def generate_response(self, prompt: str, system_prompt: str = None, max_tokens: int = 4000) -> str:
        """Generate response using Groq LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating the response."
    
    async def generate_json_response(self, prompt: str, system_prompt: str = None) -> Dict:
        """Generate structured JSON response"""
        response = await self.generate_response(prompt, system_prompt)
        
        try:
            # Extract JSON from response if wrapped in markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            return {"error": "Failed to parse JSON response", "raw_response": response}

class DatabaseInsightAgent:
    """Agent specialized in database analysis and insights"""
    
    def __init__(self, llm_manager: LLMManager, schema_extractor: DatabaseSchemaExtractor):
        self.llm_manager = llm_manager
        self.schema_extractor = schema_extractor
        self.agent_type = AgentType.DATABASE_INSIGHT
    
    async def analyze_database_structure(self, context: Dict) -> Dict:
        """Analyze database structure and generate insights"""
        schema = await self.schema_extractor.extract_complete_schema()
        
        system_prompt = """You are a database analysis expert with deep learning capabilities. Analyze the provided database schema and generate comprehensive insights about the data structure, relationships, and business context. 

        Focus on:
        1. Data quality assessment and anomaly detection
        2. Business domain identification and entity relationships
        3. Key metrics and KPIs discovery
        4. Data lineage and dependencies
        5. Potential analysis opportunities and recommendations
        6. Performance optimization suggestions
        7. Business intelligence insights
        
        Use advanced pattern recognition to identify:
        - Hidden relationships between tables
        - Business processes reflected in the data structure
        - Potential data quality issues
        - Key performance indicators
        - Growth opportunities
        
        Return structured JSON output with detailed insights and confidence scores."""
        
        prompt = f"""
        Analyze this database schema and provide deep insights:
        
        Schema Information:
        {json.dumps(schema, indent=2, default=str)}
        
        Generate comprehensive insights about:
        1. Database structure complexity and health
        2. Business domain and operational patterns
        3. Data quality indicators and issues
        4. Key tables, relationships, and data flows
        5. Performance bottlenecks and optimization opportunities
        6. Business intelligence and analytics potential
        7. Strategic recommendations for data utilization
        
        Return JSON format:
        {{
            "database_health": {{
                "overall_score": 0.85,
                "complexity_level": "medium",
                "optimization_opportunities": []
            }},
            "business_insights": {{
                "primary_domain": "e-commerce",
                "key_processes": [],
                "growth_indicators": []
            }},
            "data_quality": {{
                "issues": [],
                "recommendations": []
            }},
            "analytics_opportunities": {{
                "kpis": [],
                "trending_analysis": [],
                "predictive_potential": []
            }},
            "strategic_recommendations": []
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'analysis_type': 'database_structure',
            'schema': schema,
            'insights': response,
            'metadata': {
                'tables_count': len(schema['tables']),
                'relationships_count': len(schema['relationships']),
                'complexity_score': schema['semantic_analysis']['database_complexity'],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    async def analyze_table_data(self, table_name: str, context: Dict) -> Dict:
        """Analyze specific table data with deep learning insights"""
        table_info = context.get('schema', {}).get('tables', {}).get(table_name, {})
        
        system_prompt = """You are a data analyst expert with advanced pattern recognition capabilities. Analyze the provided table data and generate deep business insights using machine learning approaches.

        Focus on:
        1. Statistical analysis and distribution patterns
        2. Anomaly detection and outlier identification
        3. Correlation analysis and hidden relationships
        4. Trend identification and forecasting potential
        5. Business KPIs and performance metrics
        6. Data quality assessment and cleansing needs
        7. Actionable business recommendations
        
        Use advanced analytics to identify:
        - Seasonal patterns and trends
        - Customer segments and behaviors
        - Revenue optimization opportunities
        - Operational efficiency metrics
        - Risk indicators and early warning signs
        
        Provide structured analysis with confidence scores and actionable insights."""
        
        prompt = f"""
        Analyze this table data and provide deep insights:
        
        Table: {table_name}
        Information: {json.dumps(table_info, indent=2, default=str)}
        
        Generate comprehensive insights about:
        1. Data patterns, distributions, and statistical properties
        2. Business context and operational significance
        3. Quality assessment and data integrity
        4. Key findings, trends, and anomalies
        5. Performance metrics and KPIs
        6. Predictive analytics opportunities
        7. Strategic recommendations for business growth
        
        Return JSON format:
        {{
            "table_analysis": {{
                "data_quality_score": 0.92,
                "completeness": 0.98,
                "consistency": 0.95,
                "anomalies_detected": []
            }},
            "business_insights": {{
                "key_metrics": [],
                "trends": [],
                "patterns": [],
                "segments": []
            }},
            "statistical_analysis": {{
                "distributions": {{}},
                "correlations": [],
                "outliers": []
            }},
            "recommendations": {{
                "data_quality": [],
                "business_actions": [],
                "analytics_opportunities": []
            }},
            "confidence_score": 0.87
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'analysis_type': 'table_data',
            'table_name': table_name,
            'insights': response,
            'metadata': {
                'row_count': table_info.get('row_count', 0),
                'column_count': len(table_info.get('columns', [])),
                'business_domain': table_info.get('business_context', {}).get('primary_domain', 'unknown'),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

class SQLGeneratorAgent:
    """Agent specialized in SQL query generation with context awareness"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.SQL_GENERATOR
    
    async def generate_query(self, user_request: str, schema_context: Dict, session_context: Dict = None) -> Dict:
        """Generate SQL query based on user request with enhanced context"""
        system_prompt = """You are an expert SQL query generator with deep database knowledge. Generate optimized PostgreSQL queries based on user requests, database schema, and conversation context.

        Always:
        1. Use proper table and column names with double quotes
        2. Include comprehensive error handling
        3. Optimize for performance with proper indexing
        4. Add detailed comments explaining query logic
        5. Consider business context and user intent
        6. Provide query variations for different use cases
        7. Include data validation and quality checks
        
        Advanced capabilities:
        - Generate complex analytical queries
        - Create window functions for trend analysis
        - Build aggregation queries for KPIs
        - Construct joins across multiple tables
        - Implement filtering and segmentation
        - Generate time-series analysis queries
        
        Return structured JSON with query, explanation, and metadata."""
        
        schema_summary = self._create_enhanced_schema_summary(schema_context)
        context_info = self._extract_context_info(session_context) if session_context else {}
        
        prompt = f"""
        Generate advanced SQL query for this request: "{user_request}"
        
        Database Schema:
        {schema_summary}
        
        Session Context:
        {json.dumps(context_info, indent=2, default=str)}
        
        Requirements:
        1. Use PostgreSQL advanced features
        2. Include proper error handling and data validation
        3. Optimize with appropriate indexes and joins
        4. Add comprehensive comments
        5. Consider business context and KPIs
        6. Include data quality checks
        7. Provide alternative query approaches
        
        Return JSON format:
        {{
            "primary_query": {{
                "sql": "SELECT ...",
                "explanation": "Main query explanation",
                "expected_columns": [],
                "performance_notes": "Query optimization details"
            }},
            "alternative_queries": [
                {{
                    "sql": "SELECT ...",
                    "use_case": "Alternative approach",
                    "explanation": "When to use this variation"
                }}
            ],
            "validation_queries": [
                {{
                    "sql": "SELECT ...",
                    "purpose": "Data quality check"
                }}
            ],
            "metadata": {{
                "complexity": "medium",
                "estimated_execution_time": "< 1s",
                "tables_involved": [],
                "indexes_recommended": []
            }},
            "confidence_score": 0.92
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'user_request': user_request,
            'generated_queries': response,
            'metadata': {
                'schema_tables_count': len(schema_context.get('tables', {})),
                'context_relevance': len(context_info),
                'generation_timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_enhanced_schema_summary(self, schema_context: Dict) -> str:
        """Create enhanced schema summary with business context"""
        summary = []
        tables = schema_context.get('tables', {})
        relationships = schema_context.get('relationships', [])
        
        for table_name, table_info in tables.items():
            columns = table_info.get('columns', [])
            business_context = table_info.get('business_context', {})
            
            table_summary = f"Table: {table_name} ({business_context.get('primary_domain', 'general')})\n"
            table_summary += f"  Purpose: {business_context.get('table_purpose', 'unknown')}\n"
            table_summary += f"  Rows: {table_info.get('row_count', 0)}\n"
            table_summary += "  Columns:\n"
            
            for col in columns:
                pk_marker = " (PK)" if col.get('is_primary_key') else ""
                unique_marker = " (UNIQUE)" if col.get('is_unique') else ""
                nullable = "NULL" if col.get('is_nullable') == 'YES' else "NOT NULL"
                
                table_summary += f"    - {col['column_name']} {col['data_type']} {nullable}{pk_marker}{unique_marker}\n"
            
            key_metrics = business_context.get('key_metrics', [])
            if key_metrics:
                table_summary += f"  Key Metrics: {', '.join(key_metrics)}\n"
            
            summary.append(table_summary)
        
        # Add relationships summary
        if relationships:
            summary.append("\nRelationships:")
            for rel in relationships:
                summary.append(f"  {rel['table_name']}.{rel['column_name']} -> {rel['foreign_table_name']}.{rel['foreign_column_name']}")
        
        return "\n".join(summary)
    
    def _extract_context_info(self, session_context: Dict) -> Dict:
        """Extract relevant context from session"""
        return {
            'recent_queries': session_context.get('conversation_history', [])[-5:],
            'current_focus': session_context.get('current_context', {}),
            'insights_generated': len(session_context.get('insights_generated', [])),
            'session_duration': session_context.get('session_duration', 0)
        }

class VisualizationAgent:
    """Agent specialized in data visualization with deep learning insights"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.VISUALIZATION
    
    async def generate_visualization(self, query_result: Dict, insight_context: Dict, user_request: str) -> Dict:
        """Generate advanced visualization with ML-enhanced insights"""
        system_prompt = """You are a data visualization expert with deep learning capabilities. Generate comprehensive visualizations that reveal hidden patterns, trends, and insights in data.

        Capabilities:
        1. Advanced statistical visualizations (distribution plots, correlation matrices)
        2. Time-series analysis with trend detection
        3. Interactive dashboards with drill-down capabilities
        4. Business intelligence charts (KPI dashboards, performance metrics)
        5. Predictive analytics visualizations
        6. Anomaly detection charts
        7. Customer segmentation and clustering visualizations
        8. Financial analysis charts (profit/loss, ROI, growth metrics)
        
        Generate visualizations using:
        - Matplotlib for static, publication-quality charts
        - Plotly for interactive, web-based visualizations
        - Seaborn for statistical visualizations
        - Custom HTML/CSS for dashboard layouts
        
        Return structured JSON with visualization code, insights, and metadata."""
        
        data_summary = self._analyze_data_structure(query_result)
        insight_summary = self._extract_visualization_insights(insight_context)
        
        prompt = f"""
        Generate advanced visualization for this request: "{user_request}"
        
        Data Structure:
        {json.dumps(data_summary, indent=2, default=str)}
        
        Insight Context:
        {json.dumps(insight_summary, indent=2, default=str)}
        
        Requirements:
        1. Create multiple visualization types for comprehensive analysis
        2. Include interactive elements for exploration
        3. Add statistical overlays and trend lines
        4. Generate business intelligence dashboards
        5. Implement anomaly detection visualization
        6. Create predictive analytics charts
        7. Include data quality indicators
        
        Return JSON format:
        {{
            "primary_visualization": {{
                "type": "interactive_dashboard",
                "code": "import plotly.graph_objects as go...",
                "description": "Main visualization explanation",
                "insights": []
            }},
            "supporting_visualizations": [
                {{
                    "type": "statistical_analysis",
                    "code": "import matplotlib.pyplot as plt...",
                    "description": "Statistical analysis charts",
                    "insights": []
                }}
            ],
            "html_dashboard": {{
                "html_code": "<!DOCTYPE html>...",
                "css_styling": "body {{ ... }}",
                "javascript_interactions": "function updateChart() {{ ... }}"
            }},
            "insights_summary": {{
                "key_findings": [],
                "trends_detected": [],
                "anomalies": [],
                "recommendations": []
            }},
            "metadata": {{
                "visualization_complexity": "high",
                "interactivity_level": "advanced",
                "business_value": "high",
                "technical_requirements": []
            }},
            "confidence_score": 0.91
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'visualization_request': user_request,
            'generated_visualizations': response,
            'metadata': {
                'data_points': data_summary.get('row_count', 0),
                'chart_types': len(response.get('supporting_visualizations', [])) + 1,
                'interactivity_score': self._calculate_interactivity_score(response),
                'generation_timestamp': datetime.now().isoformat()
            }
        }
    
    def _analyze_data_structure(self, query_result: Dict) -> Dict:
        """Analyze data structure for visualization planning"""
        data = query_result.get('data', [])
        if not data:
            return {'row_count': 0, 'columns': [], 'data_types': {}}
        
        columns = list(data[0].keys()) if data else []
        row_count = len(data)
        
        # Analyze data types and distributions
        data_types = {}
        for col in columns:
            sample_values = [row.get(col) for row in data[:100] if row.get(col) is not None]
            if sample_values:
                if all(isinstance(v, (int, float)) for v in sample_values):
                    data_types[col] = 'numeric'
                elif all(isinstance(v, str) for v in sample_values):
                    data_types[col] = 'categorical'
                else:
                    data_types[col] = 'mixed'
        
        return {
            'row_count': row_count,
            'columns': columns,
            'data_types': data_types,
            'sample_data': data[:5]
        }
    
    def _extract_visualization_insights(self, insight_context: Dict) -> Dict:
        """Extract relevant insights for visualization"""
        return {
            'business_domain': insight_context.get('business_domain', 'general'),
            'key_metrics': insight_context.get('key_metrics', []),
            'trends': insight_context.get('trends', []),
            'anomalies': insight_context.get('anomalies', []),
            'recommendations': insight_context.get('recommendations', [])
        }
    
    def _calculate_interactivity_score(self, visualization_response: Dict) -> float:
        """Calculate interactivity score of generated visualizations"""
        score = 0.0
        
        primary_viz = visualization_response.get('primary_visualization', {})
        if 'plotly' in primary_viz.get('code', ''):
            score += 0.4
        
        if visualization_response.get('html_dashboard'):
            score += 0.3
        
        if visualization_response.get('javascript_interactions'):
            score += 0.3
        
        return min(score, 1.0)

class RecommendationAgent:
    """Agent specialized in business recommendations with deep learning insights"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.RECOMMENDATION
    
    async def generate_recommendations(self, user_query: str, context: Dict) -> Dict:
        """Generate strategic business recommendations using ML insights"""
        system_prompt = """You are a strategic business advisor with deep learning expertise. Generate actionable business recommendations based on data insights, market analysis, and predictive modeling.

        Capabilities:
        1. Strategic business planning and growth opportunities
        2. Revenue optimization and cost reduction strategies
        3. Customer segmentation and retention strategies
        4. Market trend analysis and competitive positioning
        5. Risk assessment and mitigation strategies
        6. Operational efficiency improvements
        7. Digital transformation recommendations
        8. Investment and resource allocation guidance
        
        Analysis approach:
        - Pattern recognition in business data
        - Predictive analytics for future trends
        - Competitive intelligence integration
        - ROI and impact assessment
        - Risk-reward analysis
        - Implementation roadmap planning
        
        Generate comprehensive, actionable recommendations with clear metrics and timelines."""
        
        context_summary = self._create_comprehensive_context(context)
        
        prompt = f"""
        Generate strategic business recommendations for: "{user_query}"
        
        Business Context:
        {json.dumps(context_summary, indent=2, default=str)}
        
        Requirements:
        1. Provide strategic, actionable recommendations
        2. Include predictive insights and future trends
        3. Assess risks and opportunities
        4. Suggest implementation roadmap
        5. Define success metrics and KPIs
        6. Consider market dynamics and competition
        7. Recommend resource allocation and investment
        
        Return JSON format:
        {{
            "executive_summary": {{
                "key_insights": [],
                "critical_actions": [],
                "expected_impact": "High/Medium/Low",
                "timeline": "Short/Medium/Long term"
            }},
            "strategic_recommendations": [
                {{
                    "category": "Revenue Growth",
                    "recommendation": "Specific action",
                    "rationale": "Data-driven justification",
                    "implementation_steps": [],
                    "success_metrics": [],
                    "timeline": "3-6 months",
                    "investment_required": "Low/Medium/High",
                    "expected_roi": "150%",
                    "risk_level": "Low/Medium/High"
                }}
            ],
            "predictive_insights": {{
                "trends_forecast": [],
                "market_opportunities": [],
                "potential_risks": [],
                "competitive_advantages": []
            }},
            "implementation_roadmap": {{
                "phase_1": {{
                    "duration": "0-3 months",
                    "actions": [],
                    "resources_needed": []
                }},
                "phase_2": {{
                    "duration": "3-6 months",
                    "actions": [],
                    "resources_needed": []
                }},
                "phase_3": {{
                    "duration": "6-12 months",
                    "actions": [],
                    "resources_needed": []
                }}
            }},
            "success_metrics": {{
                "kpis": [],
                "measurement_frequency": "Monthly/Quarterly",
                "target_values": {{}}
            }},
            "confidence_score": 0.88
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'user_query': user_query,
            'recommendations': response,
            'metadata': {
                'context_sources': len(context),
                'recommendation_count': len(response.get('strategic_recommendations', [])),
                'implementation_phases': len(response.get('implementation_roadmap', {})),
                'generation_timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_comprehensive_context(self, context: Dict) -> Dict:
        """Create comprehensive context for recommendations"""
        return {
            'database_insights': context.get('database_insights', {}),
            'query_results': context.get('query_results', {}),
            'visualizations': context.get('visualizations', {}),
            'business_domain': context.get('business_domain', 'general'),
            'session_history': context.get('session_history', []),
            'current_challenges': context.get('current_challenges', []),
            'business_goals': context.get('business_goals', [])
        }

class OrchestratorAgent:
    """Master orchestrator agent that coordinates all specialized agents"""
    
    def __init__(self, llm_manager: LLMManager, vector_manager: VectorStoreManager):
        self.llm_manager = llm_manager
        self.vector_manager = vector_manager
        self.agent_type = AgentType.ORCHESTRATOR
        
        # Initialize specialized agents
        self.database_agent = None
        self.sql_agent = SQLGeneratorAgent(llm_manager)
        self.visualization_agent = VisualizationAgent(llm_manager)
        self.recommendation_agent = RecommendationAgent(llm_manager)
        
        # Command patterns for different types of requests
        self.command_patterns = {
            'insight': ['analyze', 'insight', 'understand', 'explain', 'what', 'how', 'why'],
            'query': ['show', 'get', 'find', 'list', 'count', 'calculate', 'sum'],
            'visualization': ['chart', 'graph', 'plot', 'visualize', 'dashboard', 'trend'],
            'recommendation': ['recommend', 'suggest', 'advice', 'strategy', 'improve', 'optimize']
        }
    
    async def process_user_request(self, user_input: str, session_context: SessionContext) -> Dict:
        """Process user request and coordinate appropriate agents"""
        # Analyze user intent
        intent = await self._analyze_user_intent(user_input)
        
        # Handle special commands
        if user_input.startswith('/bisee'):
            return await self._handle_bisee_command(user_input, session_context)
        
        # Determine agent workflow
        workflow = await self._determine_workflow(intent, user_input, session_context)
        
        # Execute coordinated agent workflow
        result = await self._execute_workflow(workflow, user_input, session_context)
        
        # Generate comprehensive response
        response = await self._generate_comprehensive_response(result, user_input, session_context)
        
        # Store insights and update context
        await self._store_insights_and_update_context(response, session_context)
        
        return response
    
    async def _analyze_user_intent(self, user_input: str) -> Dict:
        """Analyze user intent using advanced NLP"""
        system_prompt = """You are an intent analysis expert. Analyze user queries and determine their intent, complexity, and required agents.

        Intent categories:
        1. data_insight - Understanding data patterns, relationships, quality
        2. query_generation - Creating SQL queries to extract specific data
        3. visualization - Creating charts, graphs, dashboards
        4. recommendation - Business strategy and optimization advice
        5. exploration - General data exploration and discovery
        6. comparison - Comparing different data segments or time periods
        7. prediction - Forecasting and predictive analysis
        
        Return structured analysis with confidence scores."""
        
        prompt = f"""
        Analyze this user query: "{user_input}"
        
        Determine:
        1. Primary intent and secondary intents
        2. Complexity level (simple/medium/complex)
        3. Required agents and their coordination
        4. Data requirements and scope
        5. Expected output format
        6. Business context and urgency
        
        Return JSON format:
        {{
            "primary_intent": "data_insight",
            "secondary_intents": ["visualization"],
            "complexity": "medium",
            "required_agents": ["database_insight", "visualization"],
            "agent_coordination": "sequential",
            "data_scope": "single_table",
            "output_format": "dashboard",
            "business_priority": "high",
            "confidence_score": 0.89
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        return response
    
    async def _handle_bisee_command(self, command: str, session_context: SessionContext) -> Dict:
        """Handle special /bisee commands for insight management"""
        command_parts = command.split()
        
        if len(command_parts) < 2:
            return await self._show_bisee_help()
        
        action = command_parts[1].lower()
        
        if action == 'insights':
            return await self._list_session_insights(session_context.session_id)
        elif action == 'summarize':
            insight_ids = command_parts[2:] if len(command_parts) > 2 else []
            return await self._summarize_insights(insight_ids, session_context.session_id)
        elif action == 'visualize':
            insight_ids = command_parts[2:] if len(command_parts) > 2 else []
            return await self._visualize_insights(insight_ids, session_context.session_id)
        elif action == 'recommend':
            return await self._generate_session_recommendations(session_context)
        else:
            return await self._show_bisee_help()
    
    async def _determine_workflow(self, intent: Dict, user_input: str, session_context: SessionContext) -> Dict:
        """Determine optimal agent workflow based on intent"""
        required_agents = intent.get('required_agents', [])
        complexity = intent.get('complexity', 'medium')
        
        if complexity == 'simple':
            workflow = {
                'type': 'single_agent',
                'primary_agent': required_agents[0] if required_agents else 'database_insight',
                'execution_order': required_agents
            }
        elif complexity == 'medium':
            workflow = {
                'type': 'sequential',
                'execution_order': required_agents,
                'context_passing': True
            }
        else:  # complex
            workflow = {
                'type': 'parallel_then_merge',
                'parallel_agents': required_agents[:2] if len(required_agents) > 1 else required_agents,
                'merge_agent': required_agents[-1] if len(required_agents) > 2 else 'recommendation',
                'context_sharing': True
            }
        
        return workflow
    
    async def _execute_workflow(self, workflow: Dict, user_input: str, session_context: SessionContext) -> Dict:
        """Execute the determined workflow"""
        results = {}
        
        if workflow['type'] == 'single_agent':
            agent_name = workflow['primary_agent']
            results[agent_name] = await self._execute_single_agent(agent_name, user_input, session_context)
        
        elif workflow['type'] == 'sequential':
            context = session_context.current_context
            
            for agent_name in workflow['execution_order']:
                result = await self._execute_single_agent(agent_name, user_input, session_context, context)
                results[agent_name] = result
                
                # Update context for next agent
                if workflow.get('context_passing', False):
                    context.update(result)
        
        elif workflow['type'] == 'parallel_then_merge':
            # Execute parallel agents
            parallel_tasks = []
            for agent_name in workflow['parallel_agents']:
                task = self._execute_single_agent(agent_name, user_input, session_context)
                parallel_tasks.append((agent_name, task))
            
            # Wait for parallel completion
            for agent_name, task in parallel_tasks:
                results[agent_name] = await task
            
            # Execute merge agent with all results
            merge_agent = workflow['merge_agent']
            merge_context = session_context.current_context.copy()
            merge_context.update(results)
            
            results[merge_agent] = await self._execute_single_agent(
                merge_agent, user_input, session_context, merge_context
            )
        
        return results
    
    async def _execute_single_agent(self, agent_name: str, user_input: str, 
                                   session_context: SessionContext, context: Dict = None) -> Dict:
        """Execute a single agent with proper context"""
        context = context or session_context.current_context
        
        if agent_name == 'database_insight':
            if not self.database_agent:
                # Initialize database agent with schema extractor
                schema_extractor = DatabaseSchemaExtractor(session_context.database_config)
                self.database_agent = DatabaseInsightAgent(self.llm_manager, schema_extractor)
            
            if 'analyze_structure' in user_input.lower():
                return await self.database_agent.analyze_database_structure(context)
            else:
                # Extract table name from user input or context
                table_name = self._extract_table_name(user_input, context)
                return await self.database_agent.analyze_table_data(table_name, context)
        
        elif agent_name == 'sql_generator':
            return await self.sql_agent.generate_query(user_input, context.get('schema', {}), context)
        
        elif agent_name == 'visualization':
            query_result = context.get('query_result', {})
            insight_context = context.get('insights', {})
            return await self.visualization_agent.generate_visualization(
                query_result, insight_context, user_input
            )
        
        elif agent_name == 'recommendation':
            return await self.recommendation_agent.generate_recommendations(user_input, context)
        
        else:
            return {'error': f'Unknown agent: {agent_name}'}
    
    def _extract_table_name(self, user_input: str, context: Dict) -> str:
        """Extract table name from user input or context"""
        # Simple extraction - could be enhanced with NER
        tables = list(context.get('schema', {}).get('tables', {}).keys())
        
        for table in tables:
            if table.lower() in user_input.lower():
                return table
        
        return tables[0] if tables else 'unknown'
    
    async def _generate_comprehensive_response(self, results: Dict, user_input: str, 
                                             session_context: SessionContext) -> Dict:
        """Generate comprehensive response from all agent results"""
        system_prompt = """You are a master AI assistant that synthesizes insights from multiple specialized agents. Create a comprehensive, actionable response that addresses the user's needs.

        Combine insights from:
        1. Database analysis and patterns
        2. SQL query results and data extraction
        3. Visualization and trend analysis
        4. Strategic recommendations and actions
        
        Generate a unified response that:
        - Directly answers the user's question
        - Provides actionable insights and recommendations
        - Includes relevant data visualizations
        - Suggests next steps and follow-up actions
        - Maintains business context and strategic perspective
        
        Return structured JSON with comprehensive analysis."""
        
        prompt = f"""
        User Query: "{user_input}"
        
        Agent Results:
        {json.dumps(results, indent=2, default=str)}
        
        Session Context:
        {json.dumps(asdict(session_context), indent=2, default=str)}
        
        Generate comprehensive response:
        {{
            "direct_answer": "Direct response to user question",
            "key_insights": [
                {{
                    "insight": "Key finding",
                    "evidence": "Supporting data",
                    "confidence": 0.92
                }}
            ],
            "data_analysis": {{
                "summary": "Data analysis summary",
                "key_metrics": [],
                "trends": [],
                "anomalies": []
            }},
            "visualizations": {{
                "recommended_charts": [],
                "interactive_elements": [],
                "dashboard_layout": "Description"
            }},
            "recommendations": {{
                "immediate_actions": [],
                "strategic_initiatives": [],
                "success_metrics": []
            }},
            "next_steps": [
                "Suggested follow-up actions"
            ],
            "confidence_score": 0.89
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'user_query': user_input,
            'comprehensive_response': response,
            'agent_results': results,
            'session_id': session_context.session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _store_insights_and_update_context(self, response: Dict, session_context: SessionContext):
        """Store insights and update session context"""
        # Create insight result
        insight = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type='comprehensive_analysis',
            content=response['comprehensive_response'],
            metadata={
                'agent_results': response['agent_results'],
                'user_query': response['user_query'],
                'session_id': session_context.session_id
            },
            confidence_score=response['comprehensive_response'].get('confidence_score', 0.8),
            timestamp=datetime.now()
        )
        
        # Store in vector database
        await self.vector_manager.store_insight(insight, session_context.session_id)
        
        # Update session context
        session_context.conversation_history.append({
            'query': response['user_query'],
            'response': response['comprehensive_response'],
            'timestamp': datetime.now().isoformat()
        })
        
        session_context.insights_generated.append({
            'insight_id': insight.insight_id,
            'type': insight.insight_type,
            'confidence': insight.confidence_score,
            'timestamp': insight.timestamp.isoformat()
        })
        
        # Update current context with new insights
        session_context.current_context.update({
            'last_insight': insight.content,
            'total_insights': len(session_context.insights_generated),
            'last_activity': datetime.now().isoformat()
        })
        
        # Store updated session context
        await self.vector_manager.store_session_context(session_context)
    
    async def _list_session_insights(self, session_id: str) -> Dict:
        """List all insights for a session"""
        insights = await self.vector_manager.get_session_insights(session_id)
        
        return {
            'command': 'list_insights',
            'session_id': session_id,
            'insights': [
                {
                    'id': insight['insight_id'],
                    'type': insight['insight_type'],
                    'confidence': insight['confidence_score'],
                    'timestamp': insight['timestamp'],
                    'summary': insight['content'].get('direct_answer', 'No summary available')[:100] + '...'
                }
                for insight in insights
            ],
            'total_insights': len(insights)
        }
    
    async def _summarize_insights(self, insight_ids: List[str], session_id: str) -> Dict:
        """Summarize selected insights"""
        if not insight_ids:
            all_insights = await self.vector_manager.get_session_insights(session_id)
            insight_ids = [insight['insight_id'] for insight in all_insights[-5:]]  # Last 5 insights
        
        insights_data = []
        all_insights = await self.vector_manager.get_session_insights(session_id)
        for insight_id in insight_ids:
            insight = next((i for i in all_insights if i['insight_id'] == insight_id), None)
            if insight:
                insights_data.append(insight)
        
        if not insights_data:
            return {"error": "No insights found to summarize."}

        # Generate summary using LLM
        system_prompt = """You are an insight summarization expert. Create a comprehensive summary of multiple insights, identifying patterns, trends, and key takeaways."""
        
        prompt = f"""
        Summarize these insights:
        {json.dumps(insights_data, indent=2, default=str)}
        
        Generate a comprehensive summary in this JSON format:
        {{
            "executive_summary": "High-level overview",
            "key_patterns": [],
            "recurring_themes": [],
            "data_trends": [],
            "actionable_insights": [],
            "confidence_score": 0.87
        }}
        """
        
        summary = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'command': 'summarize_insights',
            'session_id': session_id,
            'insight_ids': insight_ids,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _visualize_insights(self, insight_ids: List[str], session_id: str) -> Dict:
        """Create visualizations for selected insights"""
        if not insight_ids:
            all_insights = await self.vector_manager.get_session_insights(session_id)
            insight_ids = [insight['insight_id'] for insight in all_insights[-1:]] # Default to last insight
        
        insights_data = []
        all_insights = await self.vector_manager.get_session_insights(session_id)
        for insight_id in insight_ids:
            insight = next((i for i in all_insights if i['insight_id'] == insight_id), None)
            if insight:
                insights_data.append(insight)
        
        if not insights_data:
            return {"error": "No insights found to visualize."}

        # Use the VisualizationAgent to generate a chart based on the insight's content
        # For simplicity, we'll use the content of the first selected insight
        first_insight = insights_data[0]
        
        # The visualization agent needs data, which isn't directly in the insight.
        # We need a proxy for data. We'll use the insight summary as the context.
        visualization_request = f"Create a visualization for the following insight: {first_insight.get('content', {}).get('direct_answer', 'N/A')}"
        
        # We simulate a query result based on the insight content for the agent to use
        simulated_query_result = {
            "data": first_insight.get('content', {}).get('data_analysis', {}).get('key_metrics', []),
            "metadata": {}
        }

        visualization = await self.visualization_agent.generate_visualization(
            query_result=simulated_query_result,
            insight_context=first_insight.get('content',{}),
            user_request=visualization_request
        )

        return {
            'command': 'visualize_insights',
            'session_id': session_id,
            'insight_ids': insight_ids,
            'visualization': visualization,
            'timestamp': datetime.now().isoformat()
        }

    async def _generate_session_recommendations(self, session_context: SessionContext) -> Dict:
        """Generate recommendations based on the entire session"""
        
        # Create a comprehensive context from the entire session
        full_context = {
            'database_insights': [i for i in session_context.insights_generated if i['type'] == 'database_structure'],
            'query_results': [h for h in session_context.conversation_history if h.get('response', {}).get('data_analysis')],
            'visualizations': [h for h in session_context.conversation_history if h.get('response', {}).get('visualizations')],
            'business_domain': session_context.current_context.get('business_domain', 'general'),
            'session_history': session_context.conversation_history,
            'current_challenges': session_context.current_context.get('challenges', []),
            'business_goals': session_context.current_context.get('goals', [])
        }
        
        recommendations = await self.recommendation_agent.generate_recommendations(
            user_query="Provide overall strategic recommendations based on our entire conversation.",
            context=full_context
        )
        
        return {
            'command': 'generate_recommendations',
            'session_id': session_context.session_id,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    async def _show_bisee_help(self) -> Dict:
        """Show help for /bisee commands"""
        return {
            "command": "help",
            "message": "Available /bisee commands:",
            "commands": {
                "/bisee insights": "List all insights generated in the current session.",
                "/bisee summarize [id1] [id2]...": "Summarize specific insights by ID. Defaults to the last 5 if no IDs are provided.",
                "/bisee visualize [id]": "Generate a visualization for a specific insight. Defaults to the last one.",
                "/bisee recommend": "Generate strategic recommendations based on the entire session.",
            }
        }


async def main():
    """Main function to run the RAG chatbot"""
    # --- Configuration ---
    # Make sure to set these environment variables in a .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres")
    }

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable not set.")
        return

    # --- Initialization ---
    llm_manager = LLMManager(api_key=GROQ_API_KEY)
    vector_manager = VectorStoreManager(mongodb_uri=MONGO_URI)
    orchestrator = OrchestratorAgent(llm_manager, vector_manager)

    # --- Session Start ---
    session_id = str(uuid.uuid4())
    user_id = "test_user_123"
    
    # Create initial session context
    session_context = SessionContext(
        session_id=session_id,
        user_id=user_id,
        database_config=DB_CONFIG,
        conversation_history=[],
        insights_generated=[],
        current_context={},
        timestamp=datetime.now()
    )
    
    # Initial analysis of the database to populate context
    print("Performing initial database analysis...")
    try:
        schema_extractor = DatabaseSchemaExtractor(DB_CONFIG)
        initial_db_insight_agent = DatabaseInsightAgent(llm_manager, schema_extractor)
        initial_analysis = await initial_db_insight_agent.analyze_database_structure({})
        session_context.current_context['schema'] = initial_analysis.get('schema', {})
        session_context.current_context['database_health'] = initial_analysis.get('insights',{}).get('database_health',{})
        print("Initial analysis complete. Database schema loaded into context.")
    except Exception as e:
        logger.error(f"Could not perform initial database analysis. Please check DB connection. Error: {e}")
        # You might want to exit here if the DB is essential for the chatbot to function
        # return

    print("\n--- RAG Chatbot Initialized ---")
    print(f"Session ID: {session_id}")
    print("Type your questions about the database, or use /bisee commands. Type 'exit' to end.")

    # --- Interaction Loop ---
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Session ended. Goodbye!")
            break

        print("\nBiSee is thinking...")
        
        # Retrieve the latest session context from the database
        latest_context = await vector_manager.get_session_context(session_id)
        if latest_context:
            session_context = latest_context

        try:
            response = await orchestrator.process_user_request(user_input, session_context)
            print("\nBiSee:")
            print(json.dumps(response, indent=2, default=str))
        except Exception as e:
            logger.error(f"An error occurred while processing the request: {e}", exc_info=True)
            print("\nBiSee: I'm sorry, I encountered an unexpected error. Please try again.")

if __name__ == "__main__":
    # To run this, you would typically use:
    # asyncio.run(main())
    # This ensures the async main function is executed correctly.
    # For demonstration purposes, we will just define it. If you have a .env file set up,
    # you can uncomment the line below to run the chatbot.
    
    # Example of how to run the main loop:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting application.")

    pass