import os
from langfuse import get_client
from langfuse import observe
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Ensure environment variables are set
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-6ca23845-96c4-4884-8acb-424dac2d539b")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-64cf7b19-fbcf-42f0-8341-777ae36ac626")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-gemini-api-key")

# Configure Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Langfuse client
langfuse = get_client()

@observe()
def my_llm_call(prompt):
    """Make a call to the Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
        }
    )
    return response.text

@observe()
def my_instrumented_function(input_text):
    """Instrumented function to process input and trace with Langfuse."""
    output = my_llm_call(input_text)
    
    langfuse.update_current_trace(
        input=input_text,
        output=output,
        user_id="user_123",
        session_id="session_abc",
        tags=["agent", "my-trace"],
        metadata={"email": "user@langfuse.com"},
        version="1.0.0"
    )
    
    return output

def main():
    # Create a trace for the user interaction
    with langfuse.start_as_current_span(name="user-interaction") as span:
        user_input = input("Enter your question: ")
        span.update(input=user_input)
        
        # Call the instrumented function
        response = my_instrumented_function(user_input)
        span.update(output=response)
        
        print(f"Assistant: {response}")
        
        # Add a score for the response
        span.score_trace(
            name="user-feedback",
            value=1,
            data_type="NUMERIC",
            comment="Initial response score"
        )
        
        # Flush events to Langfuse
        langfuse.flush()

if __name__ == "__main__":
    main()