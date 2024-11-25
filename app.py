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
import smtplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import threading,json,pika,time,logging
from tools.type_string import type_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('chat.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': "chat-18938.appspot.com"
        })

# Initialize environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.your-email-provider.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "your-email@example.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# RabbitMQ configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
RABBITMQ_EXCHANGE = "notifications"
RABBITMQ_QUEUE = "email_notifications"

class EmailNotificationSystem:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.should_reconnect = True
        self.reconnect_delay = 5
        self.max_reconnect_delay = 300
        self.subscriber_thread = None
        self.logger = logging.getLogger(__name__)
        self.connection_error_logged = False  # Flag to prevent duplicate error logs

    def get_connection_params(self):
        """Get RabbitMQ connection parameters with improved error handling"""
        try:
            return pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS),
                heartbeat=600,
                connection_attempts=3,
                retry_delay=2,
                socket_timeout=5,
                # Add specific IPv4 configuration
                stack_timeout=float(5),
                tcp_options={'family': 2}  # Force IPv4
            )
        except Exception as e:
            self.logger.error(f"Error creating connection parameters: {e}")
            return None

    def initialize_rabbitmq(self):
        """Initialize RabbitMQ connection with improved error handling"""
        if not self.should_reconnect:
            return False

        try:
            # Check if RabbitMQ is actually running first
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((RABBITMQ_HOST, RABBITMQ_PORT))
            sock.close()

            if result != 0:
                if not self.connection_error_logged:
                    self.logger.error(f"RabbitMQ server is not accessible at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
                    self.logger.info("Ensure RabbitMQ server is running and accessible")
                    self.connection_error_logged = True
                return False

            # Close existing connection if any
            if self.connection and not self.connection.is_closed:
                try:
                    self.connection.close()
                except Exception:
                    pass

            # Create new connection
            params = self.get_connection_params()
            if not params:
                return False

            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            # Declare exchange and queue with error handling
            self.channel.exchange_declare(
                exchange=RABBITMQ_EXCHANGE,
                exchange_type='direct',
                durable=True
            )

            self.channel.queue_declare(
                queue=RABBITMQ_QUEUE,
                durable=True
            )

            self.channel.queue_bind(
                exchange=RABBITMQ_EXCHANGE,
                queue=RABBITMQ_QUEUE,
                routing_key='email'
            )

            self.reconnect_delay = 5  # Reset delay on successful connection
            self.connection_error_logged = False  # Reset error log flag
            self.logger.info("Successfully connected to RabbitMQ")
            return True

        except (socket.error, pika.exceptions.AMQPConnectionError) as e:
            if not self.connection_error_logged:
                self.logger.error(f"Failed to connect to RabbitMQ: {e}")
                self.connection_error_logged = True
            return self.handle_connection_failure()

        except Exception as e:
            self.logger.error(f"Unexpected error during RabbitMQ initialization: {e}")
            return self.handle_connection_failure()
    
    def process_message(self, ch, method, properties, body):
        """Process the received message and handle email notifications"""
        try:
            # Deserialize the message body (JSON)
            notification = json.loads(body)

            # Extract relevant data from the notification
            user_id = notification.get("user_id")
            email = notification.get("email")
            message_type = notification.get("message_type")
            message = notification.get("message")
            
            if not all([user_id, email, message_type, message]):
                self.logger.error("Received incomplete notification data")
                return

            # Send email notification (you can use your existing send_email method)
            subject = {
                "welcome": "Welcome to AI Assistant Chat!",
                "login": "New Login Detected"
            }.get(message_type, "AI Assistant Chat Notification")
            
            success = send_email(email, subject, message)

            if success:
                self.logger.info(f"Successfully sent email to {email}")
                ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge the message
            else:
                self.logger.error(f"Failed to send email to {email}")
                ch.basic_nack(delivery_tag=method.delivery_tag)  # Negative acknowledgment if failed

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag) 
        
    def start_subscriber(self):
        """Start the subscriber with automatic reconnection"""
        def subscription_loop():
            while self.should_reconnect:
                try:
                    if self.initialize_rabbitmq():
                        self.channel.basic_qos(prefetch_count=1)
                        self.channel.basic_consume(
                            queue=RABBITMQ_QUEUE,
                            on_message_callback=self.process_message
                        )
                        logger.info("Starting to consume messages")
                        self.channel.start_consuming()
                except Exception as e:
                    logger.error(f"Subscription error: {e}")
                    time.sleep(self.reconnect_delay)

        if not self.subscriber_thread or not self.subscriber_thread.is_alive():
            self.subscriber_thread = threading.Thread(
                target=subscription_loop, 
                daemon=True,
                name="RabbitMQ-Subscriber"
            )
            self.subscriber_thread.start()
            logger.info("Started subscriber thread")
        return self.subscriber_thread

    def handle_connection_failure(self):
        """Handle connection failures with exponential backoff"""
        if self.should_reconnect:
            self.logger.info(f"Retrying in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        return False

    def publish_notification(self, user_id, email, message_type, message):
        """Publish notification with improved error handling"""
        if not all([user_id, email, message_type, message]):
            self.logger.error("Invalid notification parameters")
            return False

        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            try:
                if not self.connection or self.connection.is_closed:
                    if not self.initialize_rabbitmq():
                        raise Exception("Failed to initialize RabbitMQ connection")

                notification = {
                    "user_id": user_id,
                    "email": email,
                    "message_type": message_type,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }

                self.channel.basic_publish(
                    exchange=RABBITMQ_EXCHANGE,
                    routing_key='email',
                    body=json.dumps(notification),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type='application/json'
                    )
                )
                self.logger.info(f"Successfully published notification for {email}")
                return True

            except Exception as e:
                current_retry += 1
                self.logger.warning(f"Attempt {current_retry} failed: {e}")
                if current_retry < max_retries:
                    time.sleep(2 ** current_retry)
                else:
                    self.logger.error(f"Failed to publish notification after {max_retries} retries")
                    # Fallback to direct email sending if RabbitMQ fails
                    return self.send_email_fallback(email, message_type, message)

    def send_email_fallback(self, recipient_email, message_type, message):
        """Fallback method to send emails directly when RabbitMQ fails"""
        try:
            subject = {
                "welcome": "Welcome to AI Assistant Chat!",
                "login": "New Login Detected"
            }.get(message_type, "AI Assistant Chat Notification")
            
            return self.send_email(recipient_email, subject, message)
        except Exception as e:
            self.logger.error(f"Email fallback failed: {e}")
            return False
# Initialize the notification system
notification_system = EmailNotificationSystem()

def send_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = formataddr(("AI Assistant Chat", EMAIL_ADDRESS))
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False
    
    
def init_db():
    database_path = os.getenv("DATABASE_PATH", "users.db")
    conn = sqlite3.connect(database_path, check_same_thread=False)
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    region TEXT,
                    email TEXT UNIQUE,
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

def signup_user(username, email, password, region):
    try:
        hashed_password = hash_password(password).decode('utf-8')
        c.execute(
            "INSERT INTO users (username, email, password, region) VALUES (?, ?, ?, ?)", 
            (username, email, hashed_password, region)
        )
        conn.commit()
        
        # Get the user_id of the newly created user
        c.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id = c.fetchone()[0]
        
        # Prepare welcome message
        welcome_message = f"""Welcome to AI Assistant Chat, {username}!
        
Thank you for joining us. Your account has been successfully created.

You can now:
- Chat with our AI Assistant
- Upload and analyze documents
- Access your chat history
- And much more!

If you have any questions, feel free to ask our AI Assistant.

Best regards,
The AI Assistant Chat Team"""
        
        # Publish welcome notification through AMPS
        notification_system.publish_notification(
            user_id, 
            email, 
            "welcome",
            welcome_message
        )
        return True
        
    except sqlite3.IntegrityError as e:
        st.error(f"Database error: {e}")
        return False
    except Exception as e:
        st.error(f"Error during signup: {e}")
        return False

def login_user(username, password):
    c.execute("SELECT user_id, password, email FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user and check_password(password, user[1]):
        user_id, _, email = user
        log_active_timestamp(user_id)
        
        # Successfully logged in without sending notifications
        return user_id
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
    search_video_tool, get_video_url_tool, get_transcript_tool, get_video_captions_tool, text_rag_tool,type_text
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
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    region = st.text_input("Region", key="region_key")  # Removed 'type' argument

    if st.button("Signup"):
        if username and email and password and region:
            if signup_user(username, email,password, region):
                st.success("Account created! Please log in.")
                st.rerun()  # Updated rerun method
            else:
                st.error("Username already exists!")
        else:
            st.warning("Please fill in all fields!")

def show_login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    email = st.text_input("Email", key="login_email")
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
            st.markdown("[Go to Admin Dashboard](http://localhost:8504)", unsafe_allow_html=True)
        else:
            st.error("Incorrect admin credentials!")

def show_chat_history():
    if st.session_state.user_id:
        history = get_user_chat_history(st.session_state.user_id)
        st.sidebar.subheader("Chat History")
        for role, message, _ in history:
            if role == "user":  # Only display messages from the user
                truncated_message = message[:50] + "..." if len(message) > 50 else message
                with st.sidebar.expander(truncated_message):
                    st.write(message)


# File upload and management
def handle_file_upload():
    # File uploader widget supporting both PDF and image file types
    uploaded_file = st.file_uploader(
        "Upload file (PDF or Image)", 
        type=["pdf", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Set up Firebase Storage bucket
        bucket = storage.bucket()

        # Generate a unique filename for the uploaded file in Firebase Storage
        filename = f"uploads/{uploaded_file.name}"

        # Determine file content type for upload
        if uploaded_file.name.lower().endswith(".pdf"):
            content_type = 'application/pdf'
        elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg")):
            content_type = f'image/{uploaded_file.name.split(".")[-1]}'
        else:
            st.error("Unsupported file type.")
            return
        
        # Create a blob (object) and upload the file to Firebase Storage
        blob = bucket.blob(filename)
        blob.upload_from_string(uploaded_file.getvalue(), content_type=content_type)

        st.success(f"File '{uploaded_file.name}' uploaded successfully to Firebase Storage.")
        
        # Save the uploaded file locally
        data_folder = "data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        # Remove all existing files with the same extension in the data folder
        extensions_to_clean = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg"]
        for file in os.listdir(data_folder):
            if any(file.endswith(ext) for ext in extensions_to_clean):
                os.remove(os.path.join(data_folder, file))
        
        # Save the newly uploaded file locally
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully to local storage.")


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
    
    notification_system.start_subscriber()
    
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