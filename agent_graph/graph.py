import json
import ast
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from agents.agents import (
    search_query_generator_agent,
    validate_search_query_agent,
    end_node 
    )
from prompts.prompts import (
    helpful_response_generator_prompt_template,
    search_query_generator_prompt_template,
    get_search_query_generator_guided_json
    )
from states.state import AgentGraphState, get_agent_graph_state, state
from langchain_core.messages import SystemMessage


def create_graph(server=None, model=None, stop=None, model_endpoint=None, question=None, column_names=None):
    graph = StateGraph(AgentGraphState)

    graph.add_node(
        "search_query_generator",
        lambda state: search_query_generator_agent(
            state=state,
            prompt=search_query_generator_prompt_template,
            feedback=lambda: state,
            model=model,
            server=server,
            guided_json=get_search_query_generator_guided_json(column_names=column_names),
            stop=stop,
            model_endpoint=model_endpoint,
            question=question,
            column_names=column_names
        )
    )

    graph.add_node(
        "validate_search_query",
        lambda state: validate_search_query_agent(
            state=state,
        )
    )

    graph.add_node("end", lambda state: end_node(state=state))

    graph.set_entry_point("search_query_generator")
    
    graph.set_finish_point("end")

    graph.add_edge("search_query_generator", "validate_search_query")

    graph.add_conditional_edges(
        "validate_search_query",
        lambda state: 
            "search_query_generator" 
            if (state["validate_search_query_response"] and 
                "error" in state["validate_search_query_response"][-1].content.lower())
            else "end"
    )

    return graph

def compile_workflow(graph):
    workflow = graph.compile()
    return workflow
