import observe
# from langfuse.openai import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import json

api_key=os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=api_key)
client= OpenAI(
    api_key= api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
messages=[]
# @observe()
def get_weather(city:str):
    url=f'https://wttr.in/{city}?format=%C+%t'
    response=requests.get(url)
    if response.status_code==200:
       return f"the weather in {city} is {response.text}"
    return "Something went wrong"

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
Available Tools:
-get_weather:Takes city name as input returns the current weather for the city
Example:
User Query:what is the weather of new york?
Output:{{"step":"plan","content":"the user is interested in weather data of new york}}
Output:{{"step""plan","content":"From the available tools I should call get_weather"}}
Output:{{"step":"action","function":"get_weather","input":"new york"}}
Output:{{"step":"observe","output":"12 degree celsius"}}
Output:{{"step":"output","The weather of new york is 12 degree celsius"}}

"""
messages.append({'role':"system",'content':System_prompt})
user_query=input("Enter your query >")
messages.append({"role":"user","content":user_query})
outputs=[]
while True:
    response=client.chat.completions.create(
        model='gemini-1.5-flash',
        messages=messages,
        response_format={"type":"json_object"},
        
    )
    output=response.choices[0].message.content
    messages.append({"role":"assistant",'content':json.dumps(output)})
    extracted_output=json.loads(output)
    if(extracted_output.get("step")=='plan'):
     print(extracted_output.get("content"))
     continue
    if(extracted_output.get("step")=='action'):
       tool_name=extracted_output.get("function") #get weather
       tool_input=extracted_output.get("input") #city name
       if(available_tools.get(tool_name,False))!=False:
          output=available_tools[tool_name].get("fn")(tool_input)
          messages.append({"role":"assistant","content":json.dumps({"step":"observe","output":output})})

    if(extracted_output.get("step")=="output"):
       print(extracted_output.get("content"))
       break