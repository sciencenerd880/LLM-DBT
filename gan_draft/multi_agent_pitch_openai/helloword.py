import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, set_default_openai_key
import openai

def load_api_key():
    # get the directory of the current file
    current_dir = Path(__file__).parent.absolute()
    
    # load the .env file
    env_path = current_dir / '.env'
    print(f"Loading .env from: {env_path}")
    
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        return None
        
    load_dotenv(dotenv_path=env_path, override=True)
    
    # get the api key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("ERROR: API key not found in .env file")
        return None
    
    # set up api keys for both sdk and direct client
    set_default_openai_key(api_key)
    openai.api_key = api_key
    os.environ['OPENAI_API_KEY'] = api_key
    
    return api_key

def test_openai_connection():
    """test if the openai api key works by making a simple call"""
    try:
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )
        
        print("OpenAI API connection successful!")
        return True
    except Exception as e:
        print(f"ERROR connecting to OpenAI API: {str(e)}")
        return False

if __name__ == "__main__":
    # test loading the api key
    api_key = load_api_key()
    
    if not api_key:
        print("Failed to load API key. Exiting.")
        sys.exit(1)
    
    print(f"API key loaded successfully: {api_key[:5]}...{api_key[-4:]}")
    
    # check if we can connect
    if not test_openai_connection():
        print("Failed to connect to OpenAI API. Please check your API key and internet connection.")
        sys.exit(1)
    
    print("Setting up agent...")
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")
    
    print("Running agent...")
    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print("\nHaiku result:")
    print(result.final_output)