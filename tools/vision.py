from llama_index.core.tools import FunctionTool
import cv2
import os
import PIL
from PIL import Image
import google.generativeai as genai

def save_and_display_image(image):
    """Save the captured image and display it."""
    save_path = "captured_image.png"
    cv2.imwrite(save_path, image)  # Save the image
    print(f"Image saved as {save_path}.")
    img = Image.open(save_path)
    img.show()  # Display the image

def process_image_and_question(image_path, question):
    """Process the image with the given question."""
    genai.configure(api_key="AIzaSyDP4qLWuFPw1IGPP3eMV7GeTC6w_J-N7I8")  # Replace with your API key
    img = PIL.Image.open(image_path)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([question, img])
    return response.text

def capture_image_and_ask_question(question):
    """Capture image from the camera and process with the question."""
    cap = cv2.VideoCapture(0)  # Open the camera
    if not cap.isOpened():
        raise Exception("Error: Could not open camera.")

    print("Capturing image in 3 seconds...")
    cv2.waitKey(3000)  # Wait for 3 seconds before capturing

    ret, frame = cap.read()  # Capture frame
    if not ret:
        raise Exception("Error: Could not read frame.")

    save_and_display_image(frame)  # Save and display the image

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close the window

    return "captured_image.png"  # Return the path to the saved image

def image_query_tool_function(query):
    """Main function to capture image and handle the question."""
    image_path = capture_image_and_ask_question(query)  # Capture image
    answer = process_image_and_question(image_path, query)  # Process image and question
    return answer  # Return the answer

# Create the FunctionTool
camera_query_tool = FunctionTool.from_defaults(
    fn=image_query_tool_function,
    name="image_query_tool",
    description="Automatically captures an image from the camera and processes it with a given question."
)
