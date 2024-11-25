import pyautogui
import time
from llama_index.core.tools import FunctionTool

def type_text_pyautogui(text: str, interval: float = 0.02):
    """
    Simulates typing the given text using pyautogui.

    Args:
        text (str): The text to type.
        interval (float): Time delay between each character (in seconds).
    """
    time.sleep(1)  # Wait 1 second to allow user to switch to the target application
    pyautogui.write(text, interval=interval)

    
type_text=FunctionTool.from_defaults(
    fn=type_text_pyautogui,
    name="type_text_tool",
    description="""This function can be used to type text into the application of the user.if the user asks for something to be typed use this function.
    EXAMPLE USE CASES:
        1.type me the code for resnet in pytorch 
        2. type me an essay on my best friend 
        3. type me a letter to my boss asking for leave 
           
    THINGS TO BE TAKEN CARE OF :
    if text is code it should be perfectly formated and intended for eg: python code should be well intended and complete 
    if text is a essay it should be properly formatted and intended
    think of the application the user wants it to be typed in and accordingly give the text
    NOTE: Only send the final text that has to be typed into the users application into the function and not any follow up questions or such""",
)