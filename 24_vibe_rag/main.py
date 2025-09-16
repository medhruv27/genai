from dotenv import load_dotenv
import asyncio
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Assuming you want to use the GenerativeModel asynchronously
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

async def call_gemini_async(prompt):
    response = await gemini_model.generate_content_async(prompt)
    return response.text

load_dotenv()

import speech_recognition as sr
from .graph import graph

messages = []


async def tts(text: str):
    response = await gemini_model.generate_content_async(text)
    # Since Gemini does not have a direct streaming TTS, we will just print the text for now.
    # If you have another TTS solution, you can integrate it here.
    print(response.text)

def main():
    r = sr.Recognizer()  # Speech to Text

    with sr.Microphone() as source:  # Mic Access
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        while True:
            print("Speak something...")
            audio = r.listen(source)

            print("Processing Audio... (STT)")
            stt = r.recognize_google(audio)

            print("You said:", stt)
            messages.append({ "role": "user", "content": stt })

            for event in graph.stream({ "messages": messages }, stream_mode="values"):
                if "messages" in event:
                    messages.append({ "role": "assistant", "content": event["messages"][-1].content })
                    event["messages"][-1].pretty_print()


# main()

asyncio.run(tts(text="Hey! (laugh) Nice to meet you. How can I help you with coding"))