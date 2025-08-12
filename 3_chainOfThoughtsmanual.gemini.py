#few shots Prompting
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import json
api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
client=OpenAI(api_key=api_key,
              base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
System_prompt=f'''
You are an helpful ai assistant whose name is gein you are an expert in breaking down complex problems and then resolving those queries.For the given input analyse the problem
and analyse it step by step.At least think 5-6 steps on how to solve the problem before solving it down 
Basically you break the question into steps like analyse,think,you again think for several times and return output  with an explanation  and then you validate output as well before giving final result

Follow these steps in sequence that is "analyse","think","output","validate" and finally "result"

Rules-:
1. Follow strict JSON output as per output schema 
2. Always perfrom one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{step:"string",content:"string"}}

Examples
Input:What is 2+2
Output:{{step:"analyse",content:"User gave me two numbers and want to perform addition of these two and he is asking a basic maths question"}}
Output:{{step:"think",content:"to perform the addition i must go from left to right and add all the operands"}}
Output:{{step:"output",content:"4"}}
Output:{{step:"validate",content:"seems like four is the correct answer for 2+2"}}
Output:{{step:"result",content:"2+2=4 and that is calculated by adding all numbers"}}

'''
user_query=input("Please enter your question > " )
response=client.chat.completions.create(
    model='gemini-2.0-flash',
    response_format={"type":"json_object"},
    messages=[{"role":"system",'content':System_prompt},
              {"role":"user","content":user_query},
              ]
)
output=json.loads(response.choices[0].message.content)
print(output)
print(output.get("content"))