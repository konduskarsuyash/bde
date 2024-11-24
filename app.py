import sqlite3
import bcrypt
import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from llama_index.core.agent import ReActAgent
from tools.realtime_search import realtime_search_tool
from tools.image_generation import generate_image_tool
from llama_index.llms.openai import OpenAI
from tools.vision import camera_query_tool
from tools.utube import search_video_tool, get_transcript_tool, get_video_captions_tool, get_video_url_tool
from tools.check_image import check_image
from tools.RAG import text_rag_tool

# firebase_config.py
import firebase_admin
from firebase_admin import credentials, storage

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('chat.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': "chat-18938.appspot.com"
        })

# Initialize environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Database setup
# Database setup with SQLite
def init_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    region TEXT,
                    password TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT,
                    role TEXT,
                    message TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS active_interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id))''')
    
    conn.commit()
    return conn, c

# Initialize database connection and cursor
conn, c = init_db()

# Function to log active timestamp
def log_active_timestamp(user_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO active_interactions (user_id, timestamp) VALUES (?, ?)", (user_id, timestamp))
    conn.commit()

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    if isinstance(hashed, str):
        hashed = hashed.encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def signup_user(username, password, region):
    try:
        hashed_password = hash_password(password).decode('utf-8')
        c.execute(
            "INSERT INTO users (username, password, region) VALUES (?, ?, ?)", 
            (username, hashed_password, region)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    c.execute("SELECT user_id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user and check_password(password, user[1]):
        return user[0]  # Return user_id
    return None

# Chat history functions
def save_message(user_id, role, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""INSERT INTO chat_history (user_id, timestamp, role, message)
                 VALUES (?, ?, ?, ?)""", 
              (user_id, timestamp, role, message))
    conn.commit()

def get_user_chat_history(user_id, limit=50):
    c.execute("""SELECT role, message, timestamp 
                 FROM chat_history 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?""", 
              (user_id, limit))
    return c.fetchall()

# Initialize the agent
system_prompt = """
You are a helpful assistant who specializes in analyzing images and generating code for neural network architectures.
If the user provides an image, examine the image and generate the code accordingly.
If the user provides a text prompt, respond appropriately based on your role.
"""

llm = OpenAI(model="gpt-4", system_prompt=system_prompt)

tools = [
    check_image, generate_image_tool, camera_query_tool, realtime_search_tool,
    search_video_tool, get_video_url_tool, get_transcript_tool, get_video_captions_tool, text_rag_tool
]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True
)

# Streamlit UI components
def show_signup():
    st.subheader("Create New Account")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    region = st.text_input("Region", key="region_key")  # Removed 'type' argument

    if st.button("Signup"):
        if username and password and region:
            if signup_user(username, password, region):
                st.success("Account created! Please log in.")
                st.rerun()  # Updated rerun method
            else:
                st.error("Username already exists!")
        else:
            st.warning("Please fill in all fields!")

def show_login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_id = user_id
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Incorrect username or password")
            

# Define admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
# Admin login function with redirection link
def admin_login():
    st.subheader("Admin Login")
    username = st.text_input("Username", key="admin_username")
    password = st.text_input("Password", type="password", key="admin_password")
    
    if st.button("Login as Admin"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.admin_logged_in = True
            st.success("Admin login successful!")
            
            # Redirect to the admin dashboard (running on localhost:8502)
            st.markdown("[Go to Admin Dashboard](http://localhost:8502)", unsafe_allow_html=True)
        else:
            st.error("Incorrect admin credentials!")

def show_chat_history():
    if st.session_state.user_id:
        history = get_user_chat_history(st.session_state.user_id)
        st.sidebar.subheader("Chat History")
        for role, message, timestamp in history:
            with st.sidebar.expander(f"{role} - {timestamp}"):
                st.write(message)

# File upload and management
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        # Set up Firebase Storage bucket
        bucket = storage.bucket()

        # Generate a unique filename for the uploaded file in Firebase Storage
        filename = f"uploads/{uploaded_file.name}"

        # Create a blob (object) and upload the file
        blob = bucket.blob(filename)
        blob.upload_from_string(uploaded_file.getvalue(), content_type='application/pdf')

        st.success(f"File '{uploaded_file.name}' uploaded successfully to Firebase Storage.")
        
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
    
    # Call the handle_file_upload function here
    st.write("Upload a PDF file for reference:")
    handle_file_upload()
    
    if st.button("Submit") and prompt:
        try:
            # Log the active timestamp whenever the user submits a query
            log_active_timestamp(st.session_state.user_id)
            
            # Handle file uploads
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
            
            save_message(st.session_state.user_id, "user", prompt)
            save_message(st.session_state.user_id, "assistant", str(response))

            # Optionally, display the agent's latest response directly
            st.write("Agent:", response_text)

            # Check for GENIMG100 code in the response
            if "GENIMG100" in response_text:
                # Display the generated image
                st.image("generated_image.png", caption="Generated Image")
            
            # Update chat history display
            show_chat_history()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_id = None
    st.success("Logged out successfully!")
    st.rerun()

# Main app
def main():
    # Initialize Firebase once
    initialize_firebase()
    st.title("AI Assistant")
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None
        if "memory" not in st.session_state:
            st.session_state.memory = []
        if "selected_question" not in st.session_state:
            st.session_state.selected_question = None
        st.session_state.admin_logged_in = False  # Track admin login state
    
    # Main app logic
    if st.session_state.logged_in:
        if st.sidebar.button("Logout"):
            logout()
        show_chat_history()
        agent_ui()
    elif st.session_state.admin_logged_in:
        # Import and run the dashboard in demo.py for admin
        from demo import run_dashboard
        run_dashboard()
    else:
        # Normal user flow
        option = st.sidebar.selectbox("Choose Action", ["User Login", "User Signup", "Admin Login"])
        if option == "User Login":
            show_login()
        elif option == "User Signup":
            show_signup()
        elif option == "Admin Login":
            admin_login()

if __name__ == "__main__":
    main()