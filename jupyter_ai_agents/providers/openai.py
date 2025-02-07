# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

import os
from typing import Union
from langchain.agents import tool, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from openai import OpenAIError

# List of valid OpenAI model names
VALID_MODELS = ["gpt-4o", "o1", "o3-mini", "gpt-4-turbo"]

def create_openai_agents(model_name: Union[str, None], system_prompt: str, tools: list) -> AgentExecutor:
    """Create an agent from a set of tools using OpenAI's API.
    
    Args:
        model_name: The name of the OpenAI model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo').
                   If None, uses the value from OPENAI_MODEL_NAME environment variable
                   or defaults to 'gpt-4o'.
        system_prompt: The system prompt to use for the agent
        tools: List of tools available to the agent
        
    Returns:
        A configured agent executor that can use the provided tools
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment variables
                  or if an invalid model name is provided
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in environment variables")
        
    # Use provided model_name, env var, or default to gpt-4o
    model_name = model_name or os.environ.get("OPENAI_MODEL_NAME") or "gpt-4o"
    if not model_name:  # Handle empty string case
        model_name = "gpt-4o"
    
    # Validate model name
    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of: {', '.join(VALID_MODELS)}")
    
    try:
        llm = ChatOpenAI(model_name=model_name)
    except OpenAIError as e:
        raise ValueError(f"Error initializing OpenAI model: {str(e)}")

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent using LangChain's built-in tool handling
    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3  # Limit iterations to prevent infinite loops
    ) 