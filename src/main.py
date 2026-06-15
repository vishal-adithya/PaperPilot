import arxiv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph,START,END
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import os
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

from langchain_core.globals import set_debug
set_debug(True)

class State(TypedDict):
    query: str
    topic: str
    fetched_papers: list[dict]
    question: str
    retrieved_docs: list[Document]

graph_builder = StateGraph(State)

def analyse_topic(state: State) -> State:
    
    prompt = PromptTemplate.from_template(
        template = """You are a research assistant. Given a user query, return a JSON with exactly these two keys:

- "topic": 3-6 space-separated keywords for arXiv search, no punctuation, no full sentences
- "question": a precise, self-contained research question answerable from academic paper abstracts. Should focus on a measurable outcome, comparison, or mechanism. Avoid vague terms like "explain" or "tell me about"

Query: {query}

Example output:
{{"topic": "RLHF LLM alignment human feedback", "question": "What measurable improvements in alignment does RLHF produce in large language models compared to supervised fine-tuning?"}}

Return ONLY the JSON, no markdown, no explanation.""")
    
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.1
            )
    
    chain = prompt | llm
    response = chain.invoke(state["query"])

    parsed = json.loads(response.content)
    return {**state, "topic": parsed["topic"], "question": parsed["question"]}
    
def fetch_papers(state: State) -> State:
    client = arxiv.Client()
    search = arxiv.Search(
        query=state["topic"],
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "url": result.pdf_url
        })
    return {**state, "fetched_papers": papers}

def rag_pipeline(state: State) -> State:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100)
    
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    doc = [
        Document(
            page_content = p["abstract"],
            metadata = {
                "title": p["title"], 
                "authors": p["authors"], 
                "url": p["url"]
                        })
            for p in state["fetched_papers"]]
    
    chunks = splitter.split_documents(doc)
    faiss = FAISS.from_documents(chunks, embed)
    
    retriever = faiss.as_retriever(
        search_kwargs = {"k": 5}
        )
    
    retrieved_docs = retriever.invoke(
        state["question"]
        )
    
    return {
            **state,
            "retrieved_docs": retrieved_docs
            }
    

if __name__ == "__main__":
    graph_builder.add_node("fetch_papers", fetch_papers)
    graph_builder.add_node("RAG_pipeline", rag_pipeline)
    graph_builder.add_node("analyse_topic", analyse_topic)
    
    graph_builder.add_edge(START, "analyse_topic")
    graph_builder.add_edge("analyse_topic", "fetch_papers")
    graph_builder.add_edge("fetch_papers","RAG_pipeline")
    graph_builder.add_edge("RAG_pipeline", END)
    
    graph = graph_builder.compile()

    display(Image(graph.get_graph().draw_mermaid_png()))

    final_state = graph.invoke({
    "query": "How does reinforcement learning from human feedback improve the alignment of large language models?"
        })
    print(final_state["topic"])
    print(final_state["question"])