import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import cassio

# Streamlit app
st.title("Document Routing System")

# Sidebar for user credentials
st.sidebar.header("Enter Your Credentials")
ASTRA_DB_APPLICATION_TOKEN = st.sidebar.text_input("Astra DB Application Token", type="password")
ASTRA_DB_ID = st.sidebar.text_input("Astra DB ID")
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

# Main app functionality
if ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_ID and GROQ_API_KEY:
    # Initialize Cassandra connection
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    # Set up embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )

    # Router setup
    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "wiki_search"] = Field(
            ...,
            description="Route the question to wikipedia or vectorstore."
        )

    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm = ChatGroq(model_name="Gemma2-9b-It")
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system_prompt = (
        "You are an expert at routing a user question to a vectorstore or wikipedia.\n"
        "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.\n"
        "Use the vectorstore for questions on these topics. Otherwise, use wiki-search."
    )
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    question_router = route_prompt | structured_llm_router

    # Wikipedia tool setup
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Graph state and nodes
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]

    def retrieve(state):
        question = state["question"]
        documents = astra_vector_store.similarity_search(question)
        return {"documents": documents, "question": question}

    def wiki_search(state):
        question = state["question"]
        docs = wiki.run(question)
        wiki_result = Document(page_content=docs)
        return {"documents": [wiki_result], "question": question}

    def route_question(state):
        question = state["question"]
        source = question_router.invoke({"question": question})
        return "wiki_search" if source.datasource == "wiki_search" else "retrieve"

    # Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    app = workflow.compile()

    # Streamlit interface
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_question:
            inputs = {
                "question": user_question,
                "generation": "",
                "documents": []
            }
            
            results = []
            for output in app.stream(inputs):
                for node_name, state_value in output.items():
                    results.append(f"Node: {node_name}")
                    if 'documents' in state_value and state_value['documents']:
                        results.append(f"Content: {state_value['documents'][0].page_content}")
            
            st.subheader("Results:")
            for result in results:
                st.write(result)
        else:
            st.warning("Please enter a question.")
else:
    st.warning("Please enter all required credentials in the sidebar to use the app.")

# Additional app information
st.sidebar.markdown("---")
st.sidebar.subheader("About This App")
st.sidebar.info(
    "This Document Routing System uses LangChain and Astra DB to intelligently route "
    "questions to either a vectorstore or Wikipedia based on the query content. "
    "It demonstrates the use of LLMs for query routing and document retrieval."
)
