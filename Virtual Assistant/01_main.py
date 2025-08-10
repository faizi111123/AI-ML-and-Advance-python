import speech_recognition as sr
import webbrowser
import pyttsx3 
from songs import songslib
import urllib.parse
import pyautogui
import time
from datetime import date
from youtubesearchpython import VideosSearch
from dotenv import dotenv_values
from hugchat import hugchat


secrets=dotenv_values('hf.env')
hf_email=secrets['EMAIL']
hf_pass=secrets['PASS']

recoginizer=sr.Recognizer()
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# A function which will act open commands
def obeycommands(command):
    if "open youtube" in command.lower():
         webbrowser.open("https://www.youtube.com/")
    elif "open google" in command.lower():
        webbrowser.open("https://www.google.com/")
    elif "open instagram" in command.lower():
        webbrowser.open("https://www.instagram.com/")
    elif "open facebook"  in command.lower():
        webbrowser.open("https://www.facebook.com/")
    elif "play on spotify" in command.lower():
        song=command.replace("play on spotify"," ")
        encoded_song_name = urllib.parse.quote(song)
    # Construct the Spotify search URL
        spotify_search_url = f"https://open.spotify.com/search/{song}"
    # Open the search URL in the default web browser
        webbrowser.open(spotify_search_url)
        time.sleep(13)  # Wait for 5 seconds
    # Coordinates where you want to click
        x, y = 761, 719
    # Move the cursor to the specified position
        pyautogui.moveTo(x, y)
        time.sleep(4)
    # Perform a click at the current cursor position
        pyautogui.click()
    elif "time" in command.lower():
        t = time.localtime()
        hour = time.strftime("%H",t)
        minute=time.strftime("%M",t)
        second=time.strftime("%S",t)
        speak(f"Current time is {hour}th hour {minute}th minute {second}th second")
    elif "play on youtube" in command.lower():
        # Search for the video on YouTube
        video_name=command.replace("play on youtube","")
        videos_search = VideosSearch(video_name, limit=1)
        results = videos_search.result()

        # Fetch the URL of the first video in the search results
        video_url = results['result'][0]['link']

        # Open the YouTube video in the default web browser
        webbrowser.open(video_url)

    elif "search about" in command.lower():
        # The query you want to search for
        query = command.replace("search about","")  # Replace with your desired search query

        # Encode the query to be URL-safe
        encoded_query = urllib.parse.quote(query)

        # Create the Google search URL
        google_search_url = f"https://www.google.com/search?q={encoded_query}"

        # Open the Google search results in the default web browser
        webbrowser.open(google_search_url)
  
    elif "today date" in command.lower():
        today = date.today()
        speak(f"The date today is {today}")
    else:
        user_input = command.lower()
        chatbot=hugchat.ChatBot(cookie_path="cookies.json")
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
        response = chatbot.chat(user_input)
        speak(response)
if __name__=="__main__":

        # obtaining audio from the microphone
        speak("Virtual assistant activated")
        while True:
            r = sr.Recognizer()
            print("recognizing...")
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    audio = r.listen(source)
                word = r.recognize_google(audio)
                print(word)
                if(word.lower()=="faizi"):
                    speak("ji")
                    with sr.Microphone() as source:
                        print("listening...")    
                        audio = r.listen(source)
                    command = r.recognize_google(audio)
                    obeycommands(command)
                elif(word.lower()=="quit"):
                    speak("Virtual assistant deactivated")
                    break    
            except Exception as e:
                print("Error; {0}".format(e))
        
