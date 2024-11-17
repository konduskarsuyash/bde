from llama_index.core.tools import FunctionTool
from PIL import Image
import PIL
import os
import google.generativeai as genai
def check_image(image):
    
    genai.configure(api_key="AIzaSyDP4qLWuFPw1IGPP3eMV7GeTC6w_J-N7I8") 
    image=get_image_from_directory(directory_path='data')
    img = PIL.Image.open(image)
    question="provided the image give a detailed description"
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([question, img])
    return response.text

import os


def get_image_from_directory(directory_path='data'):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    
    # List all files in the directory
    files = os.listdir(directory_path)

    # Find the first file with an image extension
    for file in files:
        if file.lower().endswith(image_extensions):
            return os.path.join(directory_path, file)
    
    return None

check_image = FunctionTool.from_defaults(
    fn=check_image,
    name="check_image",
    description="""this tool can be used to get the contents of an image and get detail description of the image by the agent""",
)

