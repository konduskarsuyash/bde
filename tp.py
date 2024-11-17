import sqlite3
import bcrypt
import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core.agent import ReActAgent
from tools.realtime_search import realtime_search_tool
from tools.image_generation import generate_image_tool
from llama_index.llms.openai import OpenAI
from tools.vision import camera_query_tool
from tools.utube import search_video_tool, get_transcript_tool, get_video_captions_tool, get_video_url_tool
from tools.check_image import check_image
from tools.RAG import text_rag_tool

# Initialize environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up the system prompt
system_prompt = """

You are an helpful AI assistant.
"""

# Initialize the language model with OpenAI
llm = OpenAI(model="gpt-4o", system_prompt=system_prompt)

# Define tools available for the agent
tools = [
    check_image, generate_image_tool, camera_query_tool, realtime_search_tool,
    search_video_tool, get_video_url_tool, get_transcript_tool, get_video_captions_tool, text_rag_tool
]

# Initialize the ReActAgent with tools and language model
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True
)

# Set up SQLite database for storing user credentials
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
conn.commit()

# Helper functions for authentication
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def signup_user(username, password):
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        return check_password(password, user[0])
    return False

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.memory = []  # To store conversation history
    st.session_state.selected_question = None  # Track selected question

# Signup UI
def show_signup():
    st.subheader("Create New Account")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Signup"):
        signup_user(username, password)
        st.success("Account created! Please log in.")
        st.experimental_rerun()

# Login UI
def show_login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.memory = []  # Clear conversation history on logout
    st.session_state.selected_question = None  # Clear selected question
    st.success("Logged out successfully!")
    st.experimental_rerun()

# File upload and management
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        # Remove all existing PDFs in the data folder
        for file in os.listdir(data_folder):
            if file.endswith(".pdf"):
                os.remove(os.path.join(data_folder, file))
        
        # Save the newly uploaded file
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

# Main app UI
# Main app UI
def agent_ui():
    st.title("AI Assistant")
    st.write(f"Welcome, {st.session_state.username}!")

    # Sidebar for previous questions as clickable items
    st.sidebar.title("Previous Questions")
    for i, memory in enumerate(st.session_state.memory):
        question = memory["question"]
        # Display only the first 30 characters of each question in the sidebar
        if st.sidebar.button(f"{i + 1}. {question[:30]}...", key=f"question_{i}"):
            st.session_state.selected_question = i  # Set the selected question index

    # Check if a specific question has been selected for display
    if st.session_state.selected_question is not None:
        # Display selected conversation details in the main area
        conversation = st.session_state.memory[st.session_state.selected_question]
        st.subheader("Selected Conversation")
        st.write(f"*User:* {conversation['question']}")
        st.write(f"*Agent:* {conversation['response']}")
    else:
        # Optional: Display an introductory message if no question is selected
        st.write("Click on a question in the sidebar to view the conversation here.")

    # Input box for new query
    prompt = st.text_input("Enter your query:")
    # Smaller file upload button below the text box
    st.write("Upload a PDF file for reference:")
    handle_file_upload()

    if st.button("Submit Query") and prompt:
        try:
            # Limit memory to last 10 entries and create context
            if len(st.session_state.memory) > 10:
                st.session_state.memory.pop(0)  # Remove the oldest entry
            
            # Prepare context from memory for the agent
            context = "\n".join([f"User: {m['question']}\nAgent: {m['response']}" for m in st.session_state.memory])
            full_prompt = f"{context}\nUser: {prompt}"  # Include previous context
            
            # Query the agent with the constructed prompt
            response = agent.query(full_prompt)
            response_text = response.response if hasattr(response, "response") else str(response)

            # Store the new conversation in session memory
            st.session_state.memory.append({"question": prompt, "response": response_text})

            # Optionally, display the agent's latest response directly
            st.write("Agent:", response_text)

            # Check for GENIMG100 code in the response
            if "GENIMG100" in response_text:
                # Display the generated image
                st.image("generated_image.png", caption="Generated Image")

        except Exception as e:
            st.error(f"An error occurred: {e}")



# Streamlit UI logic
st.title("AI Assistant with Login and PDF File Upload")
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        logout()
    agent_ui()
else:
    option = st.sidebar.selectbox("Choose Action", ["Login", "Signup"])
    if option == "Login":
        show_login()
    elif option == "Signup":
        show_signup()