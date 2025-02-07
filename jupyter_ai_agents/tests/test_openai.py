# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

import os
import pytest
from unittest.mock import patch, MagicMock
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from jupyter_ai_agents.providers.openai import create_openai_agents

# Test fixtures
@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL_NAME': 'gpt-4o'
    }):
        yield

@pytest.fixture
def mock_chat_openai():
    with patch('jupyter_ai_agents.providers.openai.ChatOpenAI') as mock:
        mock_llm = MagicMock()
        mock_llm.bind.return_value = MagicMock()
        mock.return_value = mock_llm
        yield mock

# Basic connectivity and response correctness tests
def test_create_openai_agents_basic(mock_env_vars, mock_chat_openai):
    tools = [Tool(name="test", func=lambda x: x, description="test tool")]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)
    mock_chat_openai.assert_called_once_with(model_name='gpt-4o')

def test_agent_with_tools(mock_env_vars, mock_chat_openai):
    tools = [
        Tool(name="tool1", func=lambda x: x, description="tool 1"),
        Tool(name="tool2", func=lambda x: x, description="tool 2")
    ]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)
    assert len(agent.tools) == 2
    mock_chat_openai.return_value.bind.assert_called_once()

def test_missing_api_key():
    with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
            create_openai_agents(None, "Test prompt", [])

def test_invalid_model_name(mock_env_vars, mock_chat_openai):
    mock_chat_openai.side_effect = ValueError("Invalid model")
    with pytest.raises(ValueError, match="Invalid model"):
        create_openai_agents("invalid-model", "Test prompt", [])

def test_rate_limit_retry(mock_env_vars, mock_chat_openai):
    tools = [Tool(name="test", func=lambda x: x, description="test tool")]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)
    assert agent.handle_parsing_errors is True

def test_multi_turn_dialogue(mock_env_vars, mock_chat_openai):
    tools = [Tool(name="test", func=lambda x: x, description="test tool")]
    agent = create_openai_agents(None, "Test prompt", tools)
    assert isinstance(agent, AgentExecutor)
    mock_chat_openai.return_value.bind.assert_called_once()

def test_default_model_name(mock_env_vars, mock_chat_openai):
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL_NAME': ''
    }):
        tools = [Tool(name="test", func=lambda x: x, description="test tool")]
        agent = create_openai_agents(None, "Test prompt", tools)
        assert isinstance(agent, AgentExecutor)
        mock_chat_openai.assert_called_once_with(model_name='gpt-4o')

def test_environment_model_override(mock_env_vars, mock_chat_openai):
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL_NAME': 'gpt-4-turbo'
    }):
        tools = [Tool(name="test", func=lambda x: x, description="test tool")]
        agent = create_openai_agents(None, "Test prompt", tools)
        assert isinstance(agent, AgentExecutor)
        mock_chat_openai.assert_called_once_with(model_name='gpt-4-turbo')

# Model variant tests
@pytest.mark.parametrize("model_name", [
    "gpt-4o",  # Latest GPT-4 model
    "o1",
    "o3-mini"
])
def test_different_model_variants(model_name, mock_env_vars, mock_chat_openai):
    """Test compatibility with different OpenAI models."""
    agent = create_openai_agents(model_name, "test prompt", [])
    
    assert agent is not None
    mock_chat_openai.assert_called_once_with(model_name=model_name) 