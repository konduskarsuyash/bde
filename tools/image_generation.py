import os
import requests
from llama_index.core.tools import FunctionTool
from openai import OpenAI
from PIL import Image
import io

def save_image_from_url(image_url, save_path):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Image saved as {save_path}.")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

def generate_image(query):
    # Use an environment variable for the API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=query,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    save_path = "generated_image.png"
    save_image_from_url(image_url, save_path)
    
    # Display the image using PIL
    display_image(image_url)

    return save_path

def display_image(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Open the image
        img = Image.open(io.BytesIO(response.content))
        img.show()  # This will pop up the image
    else:
        print(f"Failed to load image for display. Status code: {response.status_code}")

generate_image_tool = FunctionTool.from_defaults(
    fn=generate_image,
    name="generate_image",
    description="This tool can generate images using DALL-E based on the user's prompt"
)

