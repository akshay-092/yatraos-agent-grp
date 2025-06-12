# # --- This is the magic spell to fix the sqlite3 issue in some environments ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM # Using the direct CrewAI LLM wrapper

# --- RAG Imports ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Yatra OS Command Center",
    page_icon="👑",
    layout="wide"
)

# --- LLM & RAG Initialization ---

# 1. LLM for the Agents (as requested)
# This will be used by the agents for reasoning and generating responses.
llm = LLM(
    model="gemini/gemini-1.5-flash-latest",
    api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.7,
)

# 2. Embedding Model for the RAG system
# This is used ONLY to convert text to vectors for the ChromaDB memory.
# It's a separate, specialized tool for the job.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a separate, persistent ChromaDB vector store for each agent
vectorstores = {}
retrievers = {}
agent_names_list = ["Growth (Mark)", "Tech (Alex)", "Strategy (Strat)"]

for name in agent_names_list:
    # Create a unique, persistent directory for each agent's memory
    persist_directory = f"db_{name.split(' ')[0].lower()}"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings # Use the dedicated embedding model here
    )
    vectorstores[name] = vectorstore
    retrievers[name] = vectorstore.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks

# --- Agent Definitions (Our C-Suite) ---
# The agents will use the `llm` object we defined above.
agents = {
    "Growth (Mark)": Agent(
        role="Head of Growth and Marketing Co-founder",
        goal="Drive user acquisition and build a powerful brand for Yatra OS, remembering past conversations to maintain context.",
        backstory="""You are Mark, a seasoned marketing expert... You must leverage your memory of past interactions to provide coherent, context-aware advice.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    ),
    "Tech (Alex)": Agent(
        role="Principal Software Engineer and Team Lead",
        goal="Guide the entire technical development process of Yatra OS, using memory of past technical discussions to inform decisions.",
        backstory="""You are Alex, a 10-year veteran in full-stack development... You recall previous architectural decisions and discussions to ensure technical consistency.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    ),
    "Strategy (Strat)": Agent(
        role="Strategic Co-founder and CEO",
        goal="Ensure Yatra OS stays on track with its high-level vision, recalling past strategic dialogues to guide the company's future.",
        backstory="""You are Strat, a visionary co-founder... Your ability to remember and connect past strategic points is key to your role.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )
}

# --- Streamlit UI ---

st.title("👑 Yatra OS Command Center")
st.markdown("An AI-powered leadership team with persistent memory for each member.")

# --- Sidebar for Agent Selection ---
st.sidebar.title("Your AI Leadership Team")
st.sidebar.markdown("Select an agent to chat with:")

agent_names = list(agents.keys())
selected_agent_name = st.sidebar.radio(
    "AI Co-founders",
    agent_names,
    label_visibility="collapsed"
)

# Display the selected agent's info in the sidebar
st.sidebar.markdown("---")
selected_agent_instance = agents[selected_agent_name]
st.sidebar.info(f"**Role:** {selected_agent_instance.role}")
st.sidebar.markdown(f"🧠 **Memory DB:** `{vectorstores[selected_agent_name]._persist_directory}`")

# --- State Management for Conversations ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "Growth (Mark)": [{"role": "assistant", "content": "Landing page is live. What's our first move to get our first 100 signups?"}],
        "Tech (Alex)": [{"role": "assistant", "content": "The foundation is solid. What's the first feature you want to build? Let's talk database schema."}],
        "Strategy (Strat)": [{"role": "assistant", "content": "Great work on the initial setup. Let's think big picture. What will make Yatra OS a billion-dollar company?"}]
    }

# --- Main Chat Interface ---
st.subheader(f"Conversation with {selected_agent_name}")

current_chat_history = st.session_state.chat_histories[selected_agent_name]

for message in current_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Your message for {selected_agent_name}..."):
    current_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{selected_agent_name} is accessing memory and thinking..."):
            
            # --- RAG WORKFLOW ---
            # 1. RETRIEVE
            current_retriever = retrievers[selected_agent_name]
            relevant_docs = current_retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # 2. AUGMENT
            task_description = f"""
            Based on the following CONTEXT from our previous conversations, please answer the user's query.
            If the context is empty or not relevant, answer the query based on your expertise.
            
            ---
            CONTEXT:
            {context}
            ---
            USER QUERY:
            {prompt}
            ---
            
            Your final answer must be a concise, expert response that directly addresses the user's query,
            seamlessly integrating the context if it's relevant.
            """

            task = Task(
                description=task_description,
                expected_output="A concise, expert response that directly addresses the user's prompt.",
                agent=selected_agent_instance
            )
            
            crew = Crew(
                agents=[selected_agent_instance],
                tasks=[task],
                process=Process.sequential
            )
            
            try:
                # 3. GENERATE
                result = crew.kickoff()
                st.markdown(result)
                
                current_chat_history.append({"role": "assistant", "content": result})
                
                # 4. STORE
                interaction_to_store = f"User Query: {prompt}\nAgent Response: {result}"
                current_vectorstore = vectorstores[selected_agent_name]
                current_vectorstore.add_texts([interaction_to_store])
                st.toast(f"Memory updated for {selected_agent_name}!", icon="🧠")

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                current_chat_history.append({"role": "assistant", "content": f"I ran into an issue: {error_message}"})