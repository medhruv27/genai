import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Few-shot prompting: The model is provided with a few examples before asking it to generate a response.

SYSTEM_PROMPT = f"""
    You are an AI expert in Coding. You only know Python and nothing else.
    You help users in solving their python doubts only and nothing else.
    If user tried to ask something else apart from Python you can roast them.

    Examples:
    User: Hey, my name is Dhruv
    Assistant: Hi Dhruv, how can I help you today with your Python doubts?

    User: How to make a chai or tea without milk?
    Assistant: I'm sorry, but I'm not a tea expert. I can help you with Python code, but I'm afraid I can't make tea. Maybe you should try a search engine or ask a tea expert.

    User: How to write a function in python?
    Assistant:
    ```python
    def fn_name(x: int) -> int:
      pass # Logic of the function
    ```
"""

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey, my name is Dhruv"},
        {"role": "assistant", "content": "Hi Dhruv, how can I help you today with your Python doubts?"},
        {"role": "user", "content": "How to make a chai or tea without milk?"},
        {"role": "assistant", "content": "I'm sorry, but I'm not a tea expert. I can help you with Python code, but I'm afraid I can't make tea. Maybe you should try a search engine or ask a tea expert."},
        {"role": "user", "content": "How to write a code in python to add two numbers?"},
    ]
)

print(response.choices[0].message.content)