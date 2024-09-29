import json
import yaml
import os
from termcolor import colored
from models.openai_models import get_open_ai, get_open_ai_json
# from models.ollama_models import OllamaModel, OllamaJSONModel
# from models.vllm_models import VllmJSONModel, VllmModel
from prompts.prompts import (
    helpful_response_generator_prompt_template,
    search_query_generator_prompt_template,
)
from utils.helper_functions import get_current_utc_datetime, check_for_content, generate_report_from_outline
from states.state import AgentGraphState, get_agent_graph_state
from langchain_core.messages import SystemMessage
from langchain.agents import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from agent_tools.tools import ValidateSearchQueryTool


def search_query_generator_agent(state: AgentGraphState, prompt=search_query_generator_prompt_template, model=None, server=None, guided_json=None, stop=None, model_endpoint=None, question=None, column_names=None, feedback=None):
    feedback_value = feedback() if callable(feedback) else feedback
    feedback_value = check_for_content(feedback_value)

    # Process the feedback
    previous_queries = [msg.content for msg in feedback_value.get("search_query_generator_response", [])]
    validation_errors = [msg.content for msg in feedback_value.get("validate_search_query_response", [])]

    # print(previous_queries)
    # print(validation_errors)

    feedback_str = "Previous queries and validation results:\n"
    for i, (query, error) in enumerate(zip(previous_queries, validation_errors), 1):
        feedback_str += f"{i}. Query: {query}\n   Result: {error}\n\n"
    
    # print(feedback_str)

    search_query_generator_prompt = prompt.format(
        column_names=column_names,
        feedback=feedback_str
    )

    # print(search_query_generator_prompt)

    messages = [
        {"role": "system", "content": search_query_generator_prompt},
        {"role": "user", "content": f"Create the search query for this question and return it as a JSON object: {question}"}
    ]

    if server == 'openai':
        llm = get_open_ai_json(model=model)
        
    ai_msg = llm.invoke(messages)
    response = ai_msg.content

    state["search_query_generator_response"].append(SystemMessage(content=response))

    print(state)

    print(colored(f"Search Query Generator 🏭: {response}", 'magenta'))

    return state


import json

def validate_search_query_agent(state: AgentGraphState):
    try:
        latest_query = state["search_query_generator_response"][-1].content if state["search_query_generator_response"] else None
        
        if not latest_query:
            state["validate_search_query_response"].append(SystemMessage(content="No search query found to validate."))
            return state

        # Parse the JSON string to a Python dictionary
        query_dict = json.loads(latest_query)

        # Convert the modified dictionary back to a JSON string
        modified_query = json.dumps(query_dict)

        validate_tool = ValidateSearchQueryTool(modified_query)
        validation_result = validate_tool.run()

        state["validate_search_query_response"].append(SystemMessage(content=str(validation_result)))

        print(colored(f"Validator 🏭: {validation_result}", 'yellow'))

        return state
    except Exception as e:
        print(f"validate_search_query_agent: Error occurred - {str(e)}")
        import traceback
        print(traceback.format_exc())
        state["validate_search_query_response"].append(SystemMessage(content=f"Error in validation: {str(e)}"))
        return state

def end_node(state:AgentGraphState):
    state = {**state, "end_chain": "end_chain"}
    return state
