from tavily import TavilyClient
import os 
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.tools import FunctionTool
# Step 1. Instantiating your TavilyClient
tavily_client = TavilyClient(api_key=os.getenv('TAVY_API_KEY'))


def realtime_search(query):
# Step 2. Executing a simple search query
    response = tavily_client.search(query)

# Step 3. That's it! You've done a Tavily Search!
    return response['results']


realtime_search_tool=FunctionTool.from_defaults(
    fn=realtime_search,
    name="realtime_search_tool",
    description="""this tool can be used to get realtime search results. This tool can be used to get any realtime information about anything for example date, weather , current affairs ,etc""",
)