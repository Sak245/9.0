Document Routing System

Overview:
This project implements a document routing system that routes user questions to either a domain-specific vector store or a Wikipedia search. It uses a Streamlit interface to accept queries and display results. The system relies on Cassandra as the vector store (with HuggingFace embeddings for document indexing), Langchain and Langgraph for workflow management, and ChatGroq for routing decisions.

Features:
• Routes questions based on content: queries related to agents, prompt engineering, or adversarial attacks are directed to the vector store; other queries are handled via Wikipedia search.
• Uses a vector store built with Cassandra and indexed with HuggingFace embeddings.
• Implements a routing mechanism using a ChatGroq language model with structured output.
• Provides a simple and interactive Streamlit web interface.

Installation:
1. Ensure you have Python 3.7 or higher.
2. Install the required packages using pip:
   pip install langchain langgraph cassio langchain-community tiktoken langchain-groq langchainhub chromadb langchain-huggingface wikipedia streamlit

Configuration:
• Set up your API keys and tokens as environment variables or via Streamlit secrets. The application requires:
  - ASTRA_DB_APPLICATION_TOKEN (for connecting to the Astra DB via Cassio)
  - ASTRA_DB_ID (your Astra DB identifier)
  - GROQ_API_KEY (for the ChatGroq model)
• These values should be provided in the Streamlit secrets file or through another secure configuration method.

Usage:
1. Run the application with Streamlit:
   streamlit run <app_filename.py>
2. Enter your question in the provided text input.
3. The system will route your query based on its content and display the results from the vector store or Wikipedia search.

This lightweight system is designed to easily integrate document retrieval and external search capabilities in a user-friendly web application.
