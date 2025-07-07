import os
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# --- Core Dependencies ---
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
import groq
from groq import Groq
from dotenv import load_dotenv

# --- Visualization ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ENUMERATIONS AND DATA CLASSES ---

class AgentType(Enum):
    """Defines the types of specialized agents in the system."""
    DATABASE_INSIGHT = "database_insight"
    SQL_GENERATOR = "sql_generator"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    ORCHESTRATOR = "orchestrator"

class FinancialKPI(Enum):
    """
    Defines key performance indicators for the e-wallet/finance domain.
    Each KPI has a description that helps the LLM understand its business significance.
    """
    GROSS_TRANSACTION_VALUE = "Gross Transaction Value (GTV): Total monetary value of all successful transactions."
    TRANSACTION_VOLUME = "Transaction Volume: The total number of successful transactions."
    APPROVAL_RATE = "Approval Rate: The percentage of transactions that are successfully processed."
    FRAUD_RATE = "Fraud Rate: The percentage of transactions identified as fraudulent."
    AVERAGE_TRANSACTION_VALUE = "Average Transaction Value (ATV): The average monetary value per transaction."
    ACTIVE_USERS = "Active Users: The number of unique users making at least one transaction in a period (e.g., DAU, MAU)."
    NET_TAKE_RATE = "Net Take Rate: The percentage of GTV that the platform retains as revenue after fees and costs."
    CUSTOMER_LIFETIME_VALUE = "Customer Lifetime Value (CLV): The total predicted profit generated from a single customer account."

@dataclass
class SessionContext:
    """Holds the state and context for a single user session."""
    session_id: str
    user_id: str
    database_config: Dict
    conversation_history: List[Dict]
    insights_generated: List[Dict]
    current_context: Dict
    timestamp: datetime

@dataclass
class InsightResult:
    """Represents a single piece of generated insight, ready for storage."""
    insight_id: str
    insight_type: str
    content: Dict
    metadata: Dict
    confidence_score: float
    timestamp: datetime

# --- CORE SERVICES ---

class DatabaseSchemaExtractor:
    """
    Extracts and analyzes the PostgreSQL database schema with a focus on financial context.
    """
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None

    async def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            logger.info(f"Connected to database: {self.db_config.get('database')}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    async def get_schema(self) -> Dict:
        """Extracts the complete database schema, optimized for LLM consumption."""
        if not self.connection:
            await self.connect()
        
        cursor = self.connection.cursor()
        schema_info = {'tables': {}}
        
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table['table_name']
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()
            schema_info['tables'][table_name] = {
                'columns': [dict(c) for c in columns],
                'business_context': await self._infer_business_context(table_name, [c['column_name'] for c in columns])
            }
        
        cursor.close()
        return schema_info

    async def _infer_business_context(self, table_name: str, columns: List[str]) -> Dict:
        """
        Infers business context, focusing on financial and transactional keywords.
        """
        table_lower = table_name.lower()
        columns_lower = [c.lower() for c in columns]
        
        # Keywords tailored for e-wallet/finance
        financial_keywords = [
            'transaction', 'payment', 'amount', 'fee', 'balance', 'charge', 'refund', 'ewallet',
            'transfer', 'deposit', 'withdrawal', 'currency', 'merchant', 'acquirer', 'issuer'
        ]
        user_keywords = ['user', 'customer', 'client', 'account', 'profile']
        
        is_financial = any(keyword in table_lower or any(keyword in col for col in columns_lower) for keyword in financial_keywords)
        is_user_related = any(keyword in table_lower or any(keyword in col for col in columns_lower) for keyword in user_keywords)

        purpose = "general_data"
        if is_financial and 'transaction' in table_lower:
            purpose = "transaction_log"
        elif is_financial:
            purpose = "financial_entity"
        elif is_user_related:
            purpose = "user_dimension"
            
        key_metrics = [kpi.name for kpi in FinancialKPI if any(word in kpi.name.lower() for word in table_lower.split('_') + [c.split('_')[0] for c in columns_lower])]

        return {
            'primary_domain': 'finance_ewallet' if is_financial else 'general',
            'table_purpose': purpose,
            'identified_kpis': key_metrics
        }

class VectorStoreManager:
    """Manages vector storage and retrieval using MongoDB and ChromaDB."""
    def __init__(self, mongodb_uri: str, collection_name: str = "financial_insights"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_db
        self.collection = self.db[collection_name]
        self.sessions_collection = self.db.sessions
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize ChromaDB with proper settings for version 0.4.x
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db_legacy")
        except Exception:
            try:
                self.chroma_client = chromadb.EphemeralClient()
            except Exception:
                # Fallback for cases where ChromaDB fails
                self.chroma_client = None
        
        if self.chroma_client:
            try:
                self.chroma_collection = self.chroma_client.get_or_create_collection(name="financial_insights_chroma")
            except Exception:
                self.chroma_collection = None
        else:
            self.chroma_collection = None

    async def store_insight(self, insight: InsightResult, session_id: str):
        """Stores an insight in both MongoDB and ChromaDB for robust retrieval."""
        content_text = json.dumps(insight.content)
        embedding = self.encoder.encode(content_text).tolist()
        
        doc = asdict(insight)
        doc['embedding'] = embedding
        doc['session_id'] = session_id
        
        await asyncio.to_thread(self.collection.insert_one, doc)
        
        # Store in ChromaDB if available
        if self.chroma_collection:
            try:
                self.chroma_collection.add(
                    embeddings=[embedding],
                    documents=[content_text],
                    metadatas=[{'insight_id': insight.insight_id, 'session_id': session_id, 'type': insight.insight_type}],
                    ids=[insight.insight_id]
                )
            except Exception as e:
                print(f"Warning: ChromaDB storage failed: {e}")

    async def retrieve_similar_insights(self, query: str, session_id: str, limit: int = 5) -> List[Dict]:
        """Retrieves similar insights using vector search."""
        if not self.chroma_collection:
            return []
            
        try:
            query_embedding = self.encoder.encode(query).tolist()
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"session_id": session_id}
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Warning: ChromaDB query failed: {e}")
            return []

    async def store_session_context(self, context: SessionContext):
        """Saves the current session state to MongoDB."""
        await asyncio.to_thread(
            self.sessions_collection.replace_one,
            {'session_id': context.session_id},
            asdict(context),
            upsert=True
        )

    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Loads a session state from MongoDB."""
        doc = await asyncio.to_thread(self.sessions_collection.find_one, {'session_id': session_id})
        return SessionContext(**doc) if doc else None

class LLMManager:
    """Manages interactions with the Groq LLM API."""
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"

    async def generate_json_response(self, prompt: str, system_prompt: str) -> Dict:
        """Generates a structured JSON response from the LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM JSON generation failed: {e}")
            return {"error": "Failed to generate or parse LLM response."}

# --- SPECIALIZED AGENTS ---

class SQLGeneratorAgent:
    """
    Agent specialized in generating performance-optimized SQL queries
    for financial KPIs.
    """
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.SQL_GENERATOR

    async def generate_kpi_query(self, kpi: FinancialKPI, schema: Dict, context: Dict) -> Dict:
        """Generates a SQL query specifically to calculate a given financial KPI."""
        system_prompt = """
        You are an expert PostgreSQL data analyst specializing in financial services and e-wallets.
        Your task is to write a single, precise, and optimized SQL query to calculate a specific Key Performance Indicator (KPI).
        - Use the provided database schema.
        - The query must be syntactically correct for PostgreSQL.
        - Add comments to explain the logic.
        - Focus only on the requested KPI.
        - Return the result in a JSON object with the keys 'sql_query' and 'explanation'.
        """
        prompt = f"""
        Database Schema:
        {json.dumps(schema, indent=2)}

        User Context:
        {json.dumps(context, indent=2)}

        Task: Write a PostgreSQL query to calculate the following KPI:
        KPI: {kpi.name}
        Description: {kpi.value}

        Generate the JSON output.
        """
        return await self.llm_manager.generate_json_response(prompt, system_prompt)

class RecommendationAgent:
    """
    Agent that provides strategic recommendations based on the
    Descriptive, Predictive, and Prescriptive analytics framework.
    """
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.RECOMMENDATION

    async def generate_recommendations(self, analysis_results: Dict, schema: Dict) -> Dict:
        """Generates recommendations by synthesizing data analysis results."""
        system_prompt = """
        You are a top-tier business strategy consultant for a financial technology company.
        Your goal is to provide clear, actionable recommendations based on data.
        Structure your analysis using the following framework:
        1.  **Descriptive Analytics:** What happened? Summarize the key findings from the data.
        2.  **Predictive Analytics:** What will likely happen? Project future trends and identify potential opportunities or risks.
        3.  **Prescriptive Analytics:** What should we do? Provide specific, actionable recommendations to capitalize on opportunities or mitigate risks.

        Base your entire analysis on the provided data. Return a structured JSON object.
        """
        prompt = f"""
        Database Schema Context:
        {json.dumps(schema, indent=2)}

        Data Analysis Results:
        {json.dumps(analysis_results, indent=2)}

        Task: Provide a strategic recommendation based on the data, following the Descriptive, Predictive, and Prescriptive framework.
        """
        return await self.llm_manager.generate_json_response(prompt, system_prompt)

class VisualizationAgent:
    """Agent specialized in creating data visualizations with Plotly."""
    def __init__(self):
        self.agent_type = AgentType.VISUALIZATION

    async def create_kpi_dashboard(self, kpi_data: Dict) -> str:
        """Creates an HTML dashboard visualizing the KPI data using Plotly."""
        if not kpi_data:
            return "<html><body>No data to visualize.</body></html>"

        # Example: Create a simple dashboard. This can be greatly expanded.
        fig = make_subplots(rows=1, cols=len(kpi_data), subplot_titles=list(kpi_data.keys()))
        
        col_index = 1
        for kpi_name, data in kpi_data.items():
            if isinstance(data, (int, float)):
                 fig.add_trace(go.Indicator(
                    mode="number",
                    value=data,
                    title={"text": kpi_name}),
                    row=1, col=col_index
                )
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                 # Simple bar chart for list of dicts
                 df = pd.DataFrame(data)
                 if len(df.columns) >= 2:
                     fig.add_trace(go.Bar(x=df.iloc[:,0], y=df.iloc[:,1], name=kpi_name), row=1, col=col_index)

            col_index += 1
        
        fig.update_layout(title_text="Financial KPI Dashboard", showlegend=False)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- ORCHESTRATOR AGENT ---

class OrchestratorAgent:
    """
    Master agent that interprets user requests, coordinates specialized agents,
    and synthesizes results into a coherent response.
    """
    def __init__(self, llm_manager: LLMManager, vector_manager: VectorStoreManager, db_config: Dict):
        self.llm_manager = llm_manager
        self.vector_manager = vector_manager
        self.db_config = db_config
        self.agent_type = AgentType.ORCHESTRATOR
        
        self.sql_agent = SQLGeneratorAgent(llm_manager)
        self.recommendation_agent = RecommendationAgent(llm_manager)
        self.visualization_agent = VisualizationAgent()
        self.schema_extractor = DatabaseSchemaExtractor(db_config)

    async def process_user_request(self, user_input: str, session_context: SessionContext) -> Dict:
        """Main entry point for processing a user's request."""
        logger.info(f"Processing request: '{user_input}'")
        
        # 1. Get DB Schema
        schema = await self.schema_extractor.get_schema()
        session_context.current_context['schema'] = schema
        
        # 2. Identify relevant KPIs from user input
        target_kpis = self._identify_target_kpis(user_input)
        if not target_kpis:
            return {"error": "Could not identify a relevant financial KPI in your request. Please be more specific (e.g., 'analyze GTV', 'show fraud rate')."}

        # 3. Generate and Execute SQL for each KPI
        kpi_results = {}
        for kpi in target_kpis:
            logger.info(f"Generating SQL for KPI: {kpi.name}")
            query_plan = await self.sql_agent.generate_kpi_query(kpi, schema, session_context.current_context)
            
            if 'sql_query' in query_plan:
                sql = query_plan['sql_query']
                logger.info(f"Executing SQL for {kpi.name}: {sql}")
                data = await self._execute_sql(sql)
                kpi_results[kpi.name] = data
            else:
                logger.warning(f"Could not generate SQL for {kpi.name}")

        # 4. Generate Strategic Recommendations
        logger.info("Generating strategic recommendations...")
        recommendations = await self.recommendation_agent.generate_recommendations(kpi_results, schema)

        # 5. Generate Visualizations
        logger.info("Generating visualizations...")
        visualization_html = await self.visualization_agent.create_kpi_dashboard(kpi_results)

        # 6. Synthesize and Finalize Response
        final_response = {
            "user_query": user_input,
            "recommendations": recommendations,
            "visualizations_html": visualization_html,
            "raw_kpi_data": kpi_results,
            "timestamp": datetime.now().isoformat()
        }

        # 7. Update session context
        insight = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type="kpi_analysis",
            content=final_response,
            metadata={'kpis': [k.name for k in target_kpis]},
            confidence_score=0.9, # Placeholder
            timestamp=datetime.now()
        )
        await self.vector_manager.store_insight(insight, session_context.session_id)
        session_context.conversation_history.append({"role": "user", "content": user_input})
        session_context.conversation_history.append({"role": "assistant", "content": final_response})
        await self.vector_manager.store_session_context(session_context)

        return final_response

    def _identify_target_kpis(self, user_input: str) -> List[FinancialKPI]:
        """Identifies which financial KPIs are mentioned in the user's request."""
        user_input_lower = user_input.lower()
        found_kpis = []
        for kpi in FinancialKPI:
            # Check for acronym (GTV) or full name (gross transaction value)
            if kpi.name.lower() in user_input_lower or \
               ''.join(filter(str.isupper, kpi.name)) in user_input.split():
                found_kpis.append(kpi)
        return found_kpis if found_kpis else list(FinancialKPI) # Default to all if none specified

    async def _execute_sql(self, sql_query: str) -> Any:
        """Executes a SQL query and returns the result."""
        try:
            conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            # Simplify result if it's a single value
            if len(result) == 1 and len(result[0]) == 1:
                return list(result[0].values())[0]
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"SQL execution failed for query '{sql_query}': {e}")
            return {"error": str(e)}

# --- MAIN EXECUTION ---

async def main():
    """Main function to initialize and run the RAG chatbot."""
    logger.info("Initializing Financial Insights Chatbot...")

    # --- Configuration ---
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
        logger.error("FATAL: GROQ_API_KEY environment variable not set.")
        return

    # --- Initialization ---
    llm_manager = LLMManager(api_key=GROQ_API_KEY)
    vector_manager = VectorStoreManager(mongodb_uri=MONGO_URI)
    orchestrator = OrchestratorAgent(llm_manager, vector_manager, DB_CONFIG)

    # --- Session Start ---
    session_id = str(uuid.uuid4())
    user_id = "financial_analyst_01"
    
    session_context = SessionContext(
        session_id=session_id, user_id=user_id, database_config=DB_CONFIG,
        conversation_history=[], insights_generated=[], current_context={},
        timestamp=datetime.now()
    )
    await vector_manager.store_session_context(session_context)

    print("\n--- Financial Insights RAG Chatbot Initialized ---")
    print(f"Session ID: {session_id}")
    print("Ask about financial KPIs (e.g., 'Analyze GTV and Fraud Rate', 'What is our approval rate?'). Type 'exit' to end.")

    # --- Interaction Loop ---
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Session ended. Goodbye!")
            break

        print("\nBiSee is thinking...")
        
        try:
            # Retrieve the latest session context
            latest_context = await vector_manager.get_session_context(session_id)
            if latest_context:
                session_context = latest_context

            response = await orchestrator.process_user_request(user_input, session_context)
            
            print("\n--- BiSee's Analysis ---")
            # Print recommendations
            if 'recommendations' in response and response['recommendations']:
                print("\n--- Strategic Recommendations ---")
                print(json.dumps(response['recommendations'], indent=2))
            
            # Print data
            if 'raw_kpi_data' in response and response['raw_kpi_data']:
                print("\n--- KPI Data ---")
                print(json.dumps(response['raw_kpi_data'], indent=2))

            # Save and show visualization
            if 'visualizations_html' in response and response['visualizations_html']:
                file_path = f"kpi_dashboard_{uuid.uuid4()}.html"
                with open(file_path, 'w') as f:
                    f.write(response['visualizations_html'])
                print(f"\n--- Visualization ---")
                print(f"Interactive dashboard saved to: {file_path}")

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
            print("\nBiSee: I'm sorry, I encountered an unexpected error. Please check the logs and try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting application.")
