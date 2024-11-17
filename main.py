from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from tools.realtime_search import realtime_search_tool
from tools.image_generation import generate_image_tool
from llama_index.llms.openai import OpenAI
from tools.vision import camera_query_tool
from tools.utube import search_video_tool,get_transcript_tool,get_video_captions_tool,get_video_url_tool
from tools.check_image import check_image
from tools.RAG import text_rag_tool
import os
from youtube_transcript_api import YouTubeTranscriptApi
# Load environment variables and set up API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up the system prompt
system_prompt = """
You are a helpful assistant who specializes in analyzing images and generating code for neural network architectures.
If the user provides an image, examine the image and generate the code accordingly.
If the user provides a text prompt, respond appropriately based on your role.
"""

# Initialize the language model with OpenAI
llm = OpenAI(model="gpt-4", system_prompt=system_prompt)

# Define tools available for the agent
tools = [check_image, generate_image_tool, camera_query_tool, realtime_search_tool,search_video_tool,get_video_url_tool,get_transcript_tool,get_video_captions_tool,text_rag_tool]

# Custom memory to retain conversation history
memory = []

# Initialize the ReActAgent with the tools and language model
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True
)

# Start a continuous query loop
print("Starting the assistant. Type 'exit' to end the conversation.")
while True:
    # Get the user query
    prompt = input("Enter your query: ")
    
    # Check for an exit command to end the loop
    if prompt.lower() == 'exit':
        print("Ending the conversation.")
        break
    
    # Append the conversation to memory
    context = "\n".join(memory[-10:])  # Keep the last 10 interactions for context

    # Query the agent with the context-enhanced prompt
    try:
        # Pass context and prompt to agent
        full_prompt = f"{context}\nUser: {prompt}"
        response = agent.query(full_prompt)
        
        # Store the current interaction in memory
        memory.append(f"User: {prompt}")
        memory.append(f"Agent: {response}")
        
        print("Agent:", response)
    except Exception as e:
        print("An error occurred:", e)
