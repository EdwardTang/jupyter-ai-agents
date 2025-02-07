import os
import pytest
import warnings
from pydantic import PydanticDeprecatedSince20
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from openai import OpenAIError
from jupyter_ai_agents.providers.openai import create_openai_agents

# Skip these tests if no API key is provided
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

def test_real_agent_creation():
    """Test creating an agent with a real OpenAI API call."""
    tools = [Tool(name="test", func=lambda x: x, description="test tool")]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)

def test_real_tool_binding():
    """Test binding tools to the agent with a real OpenAI API call."""
    def echo(text):
        return f"Echo: {text}"
    
    tools = [
        Tool(name="echo", func=echo, description="Echoes back the input text")
    ]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "echo"

def test_real_model_override():
    """Test overriding the model name with a real OpenAI API call."""
    tools = [Tool(name="test", func=lambda x: x, description="test tool")]
    agent = create_openai_agents("gpt-4o", "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)

def test_real_error_handling():
    """Test error handling with a real OpenAI API call."""
    with pytest.raises(ValueError, match="Invalid model name"):
        create_openai_agents("completely-invalid-model", "Test prompt", [])

def test_real_multi_turn():
    """Test multi-turn dialogue with a real OpenAI API call."""
    memory = []
    
    def memory_func(text):
        memory.append(text)
        return f"Stored: {text}"
    
    def recall_func(*args, **kwargs):
        """Return the last stored information, ignoring any arguments."""
        return f"Recalled: {memory[-1] if memory else 'Nothing stored'}"
    
    tools = [
        Tool(name="store", func=memory_func, description="Store information in memory"),
        Tool(name="recall", func=recall_func, description="Recall the last stored information")
    ]
    
    agent = create_openai_agents(
        None,
        """You are a helpful assistant that can store and recall information.
        When asked to remember something, use the 'store' tool to save it.
        When asked what was stored, use the 'recall' tool to retrieve it.""",
        tools
    )
    assert isinstance(agent, AgentExecutor)
    
    # Ignore Pydantic deprecation warnings during the test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
        
        # Store information
        response = agent.invoke({
            "input": "Please remember this: Hello World"
        })
        assert isinstance(response, dict)
        assert "output" in response
        assert any(word in str(response["output"]).lower() for word in ["stored", "remembered", "saved"])
        
        # Recall information
        response = agent.invoke({
            "input": "What was the last thing I asked you to remember?"
        })
        assert isinstance(response, dict)
        assert "output" in response
        assert "Hello World" in str(response["output"]) 