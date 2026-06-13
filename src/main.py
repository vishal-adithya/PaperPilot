import arxiv
from langgraph.graph import StateGraph,START,END
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

from langchain_core.globals import set_debug
set_debug(True)

class State(TypedDict):
    query: str
    fetched_papers: list[dict]
    fais_index: Any


graph_builder = StateGraph(State)

def fetch_papers(state: State) -> State:
    client = arxiv.Client()
    search = arxiv.Search(
        query=state["query"],
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "url": result.pdf_url
        })
    return {**state, "fetched_papers": papers}

if __name__ == "__main__":
    graph_builder.add_node("fetch_papers", fetch_papers)
    graph_builder.add_edge(START, "fetch_papers")
    graph_builder.add_edge("fetch_papers", END)
    
    # Example query
    initial_state = {"query": "machine learning in healthcare"}
    
    graph = graph_builder.compile()
    final_state = graph.invoke({
    "query": "llm grooming",
    "fetched_papers": [],
    "faiss_index": None
})
    print(final_state["fetched_papers"])
