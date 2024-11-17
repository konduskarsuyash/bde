import os
import requests
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

# Define the core functions for YouTube API access
def search_video(video_title):
    """Returns the video ID and title of the first search result that matches the video title."""
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={video_title}&key={api_key}&type=video'
    response = requests.get(url)
    data = response.json()
    if data.get('items'):
        video_id = data['items'][0]['id']['videoId']
        video_title = data['items'][0]['snippet']['title']
        return video_id, video_title
    else:
        return None, None

def get_video_captions(video_id):
    """Returns the caption ID if captions are available for the video."""
    url = f'https://www.googleapis.com/youtube/v3/captions?videoId={video_id}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if data.get('items'):
        return data['items'][0]['id']
    else:
        return None

def get_transcript(youtube_video_url):
   try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

   except Exception as e:
        raise e

def get_video_url(video_id):
    """Returns the full YouTube URL for a given video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"

# Wrap each function using `FunctionTool` with partials where needed
search_video_tool = FunctionTool.from_defaults(
    fn=search_video,
    name="search_video_tool",
    description="Searches YouTube for a given video title and returns the video ID and title of the first match. If asked to find a video "
)

get_video_captions_tool = FunctionTool.from_defaults(
    fn=get_video_captions,
    name="get_video_captions_tool",
    description="Fetches the caption ID for a given video ID if captions are available.If asked to give the captions of a youtube video"
)

get_transcript_tool = FunctionTool.from_defaults(
    fn=get_transcript,
    name="get_transcript_tool",
    description="Retrieves the transcript text for a given video url.if asked to get the transcript of a youtube video"
)

get_video_url_tool = FunctionTool.from_defaults(
    fn=get_video_url,
    name="get_video_url_tool",
    description="Generates the full YouTube URL for a given video ID.If given the video id it will give the youtube url"
)
