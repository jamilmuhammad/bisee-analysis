"""
Enhanced Financial Insights Agent System - Main Application
Refactored with LangChain/LangGraph architecture and modular agents.
"""
import os
import asyncio
import logging
import uuid
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from orchestrator.workflow import FinancialInsightsOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_insights.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FinancialInsightsApp:
    """
    Main application class for the Enhanced Financial Insights Agent System.
    """
    
    def __init__(self):
        self.orchestrator = None
        self.session_id = None
        self.user_id = "financial_analyst_01"
        
    async def initialize(self):
        """Initialize the application and all services."""
        logger.info("Initializing Enhanced Financial Insights Agent System...")
        
        # Validate environment variables
        if not self._validate_environment():
            return False
        
        # Get configuration
        config = self._get_configuration()
        
        try:
            # Initialize orchestrator
            self.orchestrator = FinancialInsightsOrchestrator(
                groq_api_key=config['groq_api_key'],
                mongodb_uri=config['mongodb_uri'],
                db_config=config['db_config']
            )
            
            # Start new session
            self.session_id = str(uuid.uuid4())
            
            logger.info("System initialized successfully")
            logger.info(f"Session ID: {self.session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = [
            'GROQ_API_KEY',
            'MONGO_URI',
            'DB_HOST',
            'DB_NAME',
            'DB_USER',
            'DB_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    def _get_configuration(self) -> Dict[str, Any]:
        """Get application configuration from environment variables."""
        return {
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'mongodb_uri': os.getenv('MONGO_URI'),
            'db_config': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
        }
    
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the agent system."""
        if not self.orchestrator:
            return {
                'error': 'System not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            logger.info(f"Processing user input: '{user_input}'")
            
            # Process through orchestrator
            response = await self.orchestrator.process_request(
                user_input=user_input,
                session_id=self.session_id
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return {
                'error': f'Processing failed: {str(e)}',
                'user_input': user_input,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for the current session."""
        if not self.orchestrator:
            return {'error': 'System not initialized'}
        
        return await self.orchestrator.get_session_analytics(self.session_id)
    
    def display_response(self, response: Dict[str, Any]):
        """Display the response in a user-friendly format."""
        print("\n" + "="*80)
        print("BiSee Financial Insights Analysis")
        print("="*80)
        
        # Display errors if any
        if 'error' in response:
            print(f"\n❌ Error: {response['error']}")
            return
        
        if 'errors' in response and response['errors']:
            print("\n⚠️  Warnings:")
            for error in response['errors']:
                print(f"   • {error}")
        
        # Display KPI data
        if 'kpi_data' in response and response['kpi_data']:
            print("\n📊 KPI Analysis Results:")
            for kpi_name, data in response['kpi_data'].items():
                if isinstance(data, dict) and 'error' in data:
                    print(f"   {kpi_name}: ❌ {data['error']}")
                elif isinstance(data, (int, float)):
                    print(f"   {kpi_name}: {data:,.2f}")
                elif isinstance(data, list):
                    print(f"   {kpi_name}: {len(data)} data points")
                else:
                    print(f"   {kpi_name}: {str(data)[:100]}...")
        
        # Display recommendations
        if 'recommendations' in response and response['recommendations']:
            recommendations = response['recommendations']
            print("\n💡 Strategic Recommendations:")
            
            if 'descriptive' in recommendations:
                print("\n📈 What Happened:")
                print(f"   {recommendations['descriptive']}")
            
            if 'predictive' in recommendations:
                print("\n🔮 What Will Likely Happen:")
                print(f"   {recommendations['predictive']}")
            
            if 'prescriptive' in recommendations:
                print("\n🎯 What Should We Do:")
                if isinstance(recommendations['prescriptive'], list):
                    for i, action in enumerate(recommendations['prescriptive'], 1):
                        print(f"   {i}. {action}")
                else:
                    print(f"   {recommendations['prescriptive']}")
            
            if 'priority' in recommendations:
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                emoji = priority_emoji.get(recommendations['priority'], '⚪')
                print(f"\n{emoji} Priority Level: {recommendations['priority'].upper()}")
        
        # Display visualizations info
        if 'visualizations' in response and response['visualizations']:
            print("\n📊 Visualizations Created:")
            for viz_id, _ in response['visualizations'].items():
                filename = f"dashboard_{viz_id[:8]}.html"
                print(f"   Interactive dashboard saved as: {filename}")
                
                # Save visualization to file
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(response['visualizations'][viz_id])
                    print("   ✅ File saved successfully")
                except Exception as e:
                    print(f"   ❌ Failed to save file: {e}")
        
        # Display execution metadata
        if 'execution_metadata' in response:
            metadata = response['execution_metadata']
            print("\n⚡ Execution Summary:")
            print(f"   Time: {metadata.get('total_execution_time', 0):.2f} seconds")
            print(f"   Confidence: {metadata.get('confidence_score', 0):.2f}")
            print(f"   Agents Used: {', '.join(metadata.get('agents_executed', []))}")
    
    async def run_interactive_session(self):
        """Run an interactive session with the user."""
        if not await self.initialize():
            print("❌ Failed to initialize the system. Please check your configuration.")
            return
        
        print("\n" + "="*80)
        print("🚀 Enhanced Financial Insights Agent System")
        print("="*80)
        print(f"Session ID: {self.session_id}")
        print("\nAvailable Analysis Types:")
        print("• KPI Analysis (e.g., 'Show me GTV and fraud rate')")
        print("• Trend Analysis (e.g., 'Analyze transaction trends over time')")
        print("• Dashboard Creation (e.g., 'Create a dashboard with all KPIs')")
        print("• Custom Queries (e.g., 'What's our approval rate by merchant?')")
        print("\nSpecial Commands:")
        print("• 'analytics' - Show session analytics")
        print("• 'help' - Show available commands")
        print("• 'exit' - End session")
        print("\nType your question below:")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("\n👋 Session ended. Thank you for using BiSee!")
                    break
                
                elif user_input.lower() == 'analytics':
                    print("\n📊 Getting session analytics...")
                    analytics = await self.get_session_analytics()
                    if 'error' not in analytics:
                        print(f"Analytics: {analytics}")
                    else:
                        print(f"❌ {analytics['error']}")
                    continue
                
                elif user_input.lower() == 'help':
                    print("\n📚 Help - Available Commands:")
                    print("• Ask about specific KPIs: 'What is our GTV?'")
                    print("• Request comparisons: 'Compare fraud rate vs approval rate'")
                    print("• Ask for trends: 'Show transaction volume trends'")
                    print("• Request dashboards: 'Create a comprehensive dashboard'")
                    print("• Get recommendations: 'What actions should we take?'")
                    continue
                
                print("\n🤖 BiSee is analyzing...")
                
                # Process the request
                response = await self.process_user_input(user_input)
                
                # Display the response
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in interactive session: {e}")
                print(f"\n❌ An unexpected error occurred: {e}")


async def main():
    """Main function to run the Financial Insights App."""
    app = FinancialInsightsApp()
    await app.run_interactive_session()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
