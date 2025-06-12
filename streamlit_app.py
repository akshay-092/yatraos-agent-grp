
# --- This is the magic spell to fix the sqlite3 issue ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM # Using the direct CrewAI LLM wrapper

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Yatra OS Command Center",
    page_icon="👑",
    layout="wide"
)

# --- LLM Initialization (Using the CrewAI LLM wrapper) ---
llm = LLM(
    model="gemini/gemini-1.5-flash-latest",
    api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.7,
)

# --- Agent Definitions (Our C-Suite) ---
agents = {
    "Growth (Mark)": Agent(
        role="Head of Growth and Marketing Co-founder",
        goal="Drive user acquisition and build a powerful brand for Yatra OS.",
        backstory="""You are Mark, a seasoned marketing expert known for taking SaaS startups
        from zero to hero. Your strategies are scrappy, data-driven, and focused on growth.
        You excel at content marketing and building a brand in public.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    ),
    "Tech (Alex)": Agent(
        role="Principal Software Engineer and Team Lead",
        goal="Guide the entire technical development process of Yatra OS.",
        backstory="""You are Alex, a 10-year veteran in full-stack development. You master modern
        web tech like Next.js and Firebase. You don't just write code; you design scalable,
        secure, and high-performance systems. You are the architect of the Yatra OS platform.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    ),
    "Strategy (Strat)": Agent(
        role="Strategic Co-founder and CEO",
        goal="Ensure Yatra OS stays on track with its high-level vision and makes smart business decisions.",
        backstory="""You are Strat, a visionary co-founder with vast experience building million-dollar
        startups. You think about the big picture: market positioning, long-term product strategy,
        fundraising narratives, and competitive moats. You are the strategic sounding board.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )
}

# --- Streamlit UI ---

st.title("👑 Yatra OS Command Center")

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


# --- State Management for Conversations ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "Growth (Mark)": [{"role": "assistant", "content": "Landing page is live. What's our first move to get our first 100 signups?"}],
        "Tech (Alex)": [{"role": "assistant", "content": "The foundation is solid. What's the first feature you want to build? Let's talk database schema."}],
        "Strategy (Strat)": [{"role": "assistant", "content": "Great work on the initial setup. Let's think big picture. What will make Yatra OS a billion-dollar company?"}]
    }

# --- Main Chat Interface ---
st.subheader(f"Conversation with {selected_agent_name}")

# Get the chat history for the currently selected agent
current_chat_history = st.session_state.chat_histories[selected_agent_name]

# Display chat messages
for message in current_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Your message for {selected_agent_name}..."):
    # Add user message to the correct chat history
    current_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{selected_agent_name} is thinking..."):
            task = Task(
                description=prompt,
                expected_output="A concise, expert response that directly addresses the user's prompt.",
                agent=selected_agent_instance
            )
            crew = Crew(
                agents=[selected_agent_instance],
                tasks=[task],
                process=Process.sequential
            )
            try:
                result = crew.kickoff()
                st.markdown(result)
                # Add agent response to the correct chat history
                current_chat_history.append({"role": "assistant", "content": result})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                current_chat_history.append({"role": "assistant", "content": f"I ran into an issue: {error_message}"})
