from litellm_client import LiteLLMClient

if __name__ == "__main__":
    #Iniatialize litellm client & define model
    client = LiteLLMClient()
    selected_model = "gpt-4"

    # Define the system prompt, task prompt, and formatting messages
    system_prompt = "You are a funny AI that speaks like Donald Trump."
    user_input = "How are you?"    
    messages = [{"role": "user", "content": user_input}]

    # Get response using litellm client
    try:
        response = client.generate_response(selected_model, 
                                            messages, 
                                            system_prompt)
        print("\nResponse:\n", response)
    except ValueError as e:
        print(f"Error: {e}")