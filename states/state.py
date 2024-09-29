from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class AgentGraphState(TypedDict):
    query_question: str
    search_query_generator_response: Annotated[list, add_messages]
    validate_search_query_response: Annotated[list, add_messages]

# Update your initial state
state = {
    "search_query_generator_response": [],
    "validate_search_query_response": []
}

# Update get_agent_graph_state function
def get_agent_graph_state(state: AgentGraphState, state_key: str):
    if state_key == "search_query_generator_all":
        return state["search_query_generator_response"]
    elif state_key == "search_query_generator_latest":
        if state["search_query_generator_response"]:
            return state["search_query_generator_response"][-1]
        else:
            return state["search_query_generator_response"]
    
    elif state_key == "validate_search_query_all":
        return state["validate_search_query_response"]
    elif state_key == "validate_search_query_latest":
        if state["validate_search_query_response"]:
            return state["validate_search_query_response"][-1]
        else:
            return state["validate_search_query_response"]

state = {
    "query_question": "",
    "search_query_generator_response": [],
    "validate_search_query_response": []
}
