import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Ensure "GOOGLE_API_KEY" matches the name in your .env file.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# You can keep your tool definition for reference in the prompt
def get_weather(city: str):
    """A dummy function for demonstration."""
    return "31 degrees"

available_tools = {
    'get_weather': {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather of the city"
    }
}

# The system prompt is well-defined and does not need changes.
System_prompt = f"""You are a powerful AI agent who is specialized in resolving user queries.
 You work on a start, plan, action, observe mode.
For the given user query and available tools, plan step-by-step execution. Based on the planning, select the relevant tools from the available
tools. Based on the tool selection, you perform an action to call the tool, wait for observation, and based on the observation from the tool call, resolve the user query.

Rules:
- Follow the output JSON format.
- Always perform one step at a time and wait for the next input.
- Carefully analyze the user query.

Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "function": "the name of the function if the step is action",
    "input": "the input parameter for the function"
}}

Example:
User Query: what is the weather of new york?
Output: {{"step":"plan","content":"The user is interested in weather data of new york"}}
Output: {{"step":"plan","content":"From the available tools I should call get_weather"}}
Output: {{"step":"action","function":"get_weather","input":"new york"}}
Output: {{"step":"observe","output":"12 degree celsius"}}
Output: {{"step":"output","content":"The weather of new york is 12 degree celsius"}}
"""

# --- Model Initialization ---
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=System_prompt
)


# --- CORRECTED: Conversation History ---
# The key 'content' is replaced with 'parts', and the value is a list.
contents = [
    {'role': 'user', 'parts': ["weather of new York"]},
    {'role': 'model', 'parts': [json.dumps({"step": "plan","content": "The user is interested in the weather data of New York."})]},
    {'role': 'model', 'parts': [json.dumps({"step": "plan", "content": "From the available tools, I should call get_weather."})]},
    {'role': 'model', 'parts': [json.dumps({"step": "action", "function": "get_weather", "input": "new york"})]},
    {'role': 'model', 'parts': [json.dumps({"step": "observe", "output": "15 degree celsius, clear sky"})]}
]

# --- JSON Mode Configuration ---
generation_config = {
    "response_mime_type": "application/json",
}

# --- API Call ---
response = model.generate_content(
    contents,
    generation_config=generation_config
)

# --- Printing the Response ---
print(response.text)