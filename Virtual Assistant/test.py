# import webbrowser
# import urllib.parse
# import pyautogui
# import time

# # Name of the song to search for
# song_name = "Shape of You"

# # Encode the song name to be URL-friendly
# encoded_song_name = urllib.parse.quote(song_name)

# # Construct the Spotify search URL
# spotify_search_url = f"https://open.spotify.com/search/{encoded_song_name}"

# # Open the search URL in the default web browser
# webbrowser.open(spotify_search_url)

# # Wait for the page to load (adjust the time as needed)
# time.sleep(5)  # Wait for 5 seconds

# # Coordinates where you want to click
# x, y = 1655, 747

# # Move the cursor to the specified position
# pyautogui.moveTo(x, y)

# # Perform a click at the current cursor position
# pyautogui.click()

# import webbrowser
# from youtubesearchpython import VideosSearch

# # The name of the video you want to search for
# video_name = "Never Gonna Give You Up"  # Replace with your desired video name

# # Search for the video on YouTube
# videos_search = VideosSearch(video_name, limit=1)
# results = videos_search.result()

# # Fetch the URL of the first video in the search results
# video_url = results['result'][0]['link']

# # Open the YouTube video in the default web browser
# webbrowser.open(video_url)

from dotenv import dotenv_values
from hugchat import hugchat
from hugchat.login import Login

secrets = dotenv_values('hf.env')

hf_email = secrets['EMAIL']
hf_pass = secrets['PASS']


     

# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)
     

prompt = "What is Streamlit?"
response = generate_response(prompt, hf_email, hf_pass)
print(response)