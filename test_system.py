"""
Basic test runner for the Enhanced Financial Insights Agent System.
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_system_initialization():
    """Test basic system initialization."""
    print("🧪 Testing System Initialization...")
    
    try:
        from orchestrator.workflow import FinancialInsightsOrchestrator
        
        # Mock configuration for testing
        test_config = {
            'groq_api_key': os.getenv('GROQ_API_KEY', 'test_key'),
            'mongodb_uri': os.getenv('MONGO_URI', 'mongodb://localhost:27017/'),
            'db_config': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'test_db'),
                'user': os.getenv('DB_USER', 'test_user'),
                'password': os.getenv('DB_PASSWORD', 'test_password')
            }
        }
        
        # Initialize orchestrator
        orchestrator = FinancialInsightsOrchestrator(
            groq_api_key=test_config['groq_api_key'],
            mongodb_uri=test_config['mongodb_uri'],
            db_config=test_config['db_config']
        )
        
        print("✅ Orchestrator initialized successfully")
        
        # Test agent initialization
        assert orchestrator.router_agent is not None
        assert orchestrator.sql_agent is not None
        assert orchestrator.recommendation_agent is not None
        assert orchestrator.visualization_agent is not None
        
        print("✅ All agents initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False

async def test_agent_types():
    """Test core type definitions."""
    print("🧪 Testing Core Types...")
    
    try:
        from core.types import AgentType, TaskType, FinancialKPI, AgentState
        
        # Test enums
        assert len(AgentType) > 0
        assert len(TaskType) > 0
        assert len(FinancialKPI) > 0
        
        print("✅ Core enums loaded successfully")
        
        # Test AgentState
        state = AgentState(
            user_input="Test input",
            session_id="test_session"
        )
        
        assert state.user_input == "Test input"
        assert state.session_id == "test_session"
        
        print("✅ AgentState working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Core types test failed: {e}")
        return False

async def test_model_configurations():
    """Test model configuration management."""
    print("🧪 Testing Model Configurations...")
    
    try:
        from core.models import ModelManager
        
        # Test model configurations
        router_config = ModelManager.get_model_config("router")
        sql_config = ModelManager.get_model_config("sql_generator")
        
        assert router_config.model_name is not None
        assert sql_config.model_name is not None
        assert router_config.cost_per_token < sql_config.cost_per_token  # Router should be cheaper
        
        print("✅ Model configurations loaded successfully")
        
        # Test cost estimation
        cost = ModelManager.estimate_cost("router", 1000)
        assert cost > 0
        
        print("✅ Cost estimation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

async def test_prompt_templates():
    """Test few-shot prompting templates."""
    print("🧪 Testing Prompt Templates...")
    
    try:
        from core.prompts import PromptTemplates
        from core.types import FinancialKPI
        
        # Test router prompt generation
        router_prompt = PromptTemplates.get_router_prompt("Show me GTV analysis")
        assert len(router_prompt) > 100  # Should include examples
        assert "KPI_ANALYSIS" in router_prompt
        
        print("✅ Router prompt generation working")
        
        # Test SQL prompt generation
        mock_schema = {"tables": {"transactions": {"columns": []}}}
        sql_prompt = PromptTemplates.get_sql_prompt(
            FinancialKPI.GROSS_TRANSACTION_VALUE,
            mock_schema,
            {}
        )
        assert "PostgreSQL" in sql_prompt
        assert "GTV" in sql_prompt or "Gross Transaction Value" in sql_prompt
        
        print("✅ SQL prompt generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt templates test failed: {e}")
        return False

async def run_all_tests():
    """Run all basic tests."""
    print("🚀 Starting Enhanced Financial Insights Agent System Tests")
    print("=" * 60)
    
    tests = [
        test_agent_types,
        test_model_configurations,
        test_prompt_templates,
        test_system_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check configuration and dependencies.")
        print("\nTroubleshooting:")
        print("1. Ensure all environment variables are set in .env file")
        print("2. Check database and MongoDB connectivity")
        print("3. Verify Groq API key is valid")
        print("4. Install all required dependencies: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
