from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import json
from langsmith.wrappers import wrap_openai
from langsmith import traceable
api_key=os.getenv("OPENAI_API_KEY")
client=wrap_openai(OpenAI(api_key=api_key))
messages=[]
def get_weather(city:str):
    return "31 degree"

available_tools={
    'get_weather':{
        "fn":get_weather,
        "description":"Takes a city name as an input and return current weather of the city"
    }
}

System_prompt=f"""You are a powerful ai agent who is specialized in resolving user query.
 You work on start,plan,action,observe mode.
For the given user query and available tools plan step by step execution,based on the planning select the relevant tools from the avilable 
tools and based on the tool selction you perfom an action to call the tool wait for observation and based on observation from the tool call resolve the user query

Rules-:
- Follow the output JSON format.
- Always Perform one step at a time and wait for next input.
- Carefully analyse the user query

Output JSON Format:
{{
    "step":"string",
"content":"string",
"function":"the name of function if the step is action",
"input":"the input parameter for the function"
}}

Example:
User Query:what is the weather of new york?
Output:{{"step":"plan","content":"the user is interested in weather data of new york}}
Output:{{"step""plan","content":"From the available tools I should call get_weather"}}
Output:{{"step":"action","function":"get_weather","input":"new york"}}
Output:{{"step":"observe","output":"12 degree celsius"}}
Output:{{"step":"output","The weather of new york is 12 degree celsius"}}

"""
response=client.chat.completions.create(
    model='gpt-4o',
    messages=[{
        'role':"system",'content':System_prompt},
        {'role':'user','content':"weather of new York"
    },
    {'role':'assistant','content':json.dumps({
    "step": "plan",
    "content": "The user is interested in the weather data of New York."
})},
  {'role':'assistant','content':json.dumps({
      "step": "plan", "content": "From the available tools, I should call get_weather."
  })},
   {'role':'assistant','content':json.dumps({
     "step": "action", "function": "get_weather", "input": "new york"
  })},
  {'role':'assistant','content':json.dumps({
     "step": "observe", "output": "15 degree celsius, clear sky"
  })}



],
response_format={"type":"json_object"}
)

print(response.choices[0].message.content)