import os
import asyncio
import logging
from agents import Agent, Runner
from utils import load_api_key, setup_logging, get_log_level, format_agent_response

async def main():
    # set up logging
    logger = setup_logging(logging.DEBUG)
    logger.info("Starting handoff test")
    
    # load API key
    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    
    # create a specialist agent
    logger.info("Creating specialist agent")
    specialist = Agent(
        name="Pitch Specialist",
        handoff_description="Expert in creating pitches for SharkTank",
        instructions="""
        You are a specialist in creating pitches for SharkTank.
        Provide detailed advice on how to pitch a product effectively.
        Include information about:
        1. How to structure the pitch
        2. Key elements to include
        3. Common mistakes to avoid
        4. Tips for handling questions
        """,
        model="gpt-4o"
    )
    
    # create a router agent that directs questions to specialists
    logger.info("Creating router agent")
    router = Agent(
        name="Pitch Router",
        handoff_description="Routes questions to appropriate specialists",
        instructions="""
        You are a router that directs questions to appropriate specialists.
        If the user asks about pitching a product on SharkTank, hand off to the Pitch Specialist.
        Otherwise, answer the question yourself.
        """,
        model="gpt-4o",
        handoffs=[specialist]
    )
    
    query = "I'm going to be on SharkTank next month. How should I pitch my product?"
    logger.info(f"Running agent with query: {query}")
    
    try:
        result = await Runner.run(router, query)
        logger.info("Agent execution completed successfully")
        logger.debug(format_agent_response("Router", result.final_output))
        print("\n=== RESULT ===\n")
        print(result.final_output)
        
        # log debug info
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Available attributes: {dir(result)}")
        
        if hasattr(result, 'metadata'):
            logger.debug(f"Metadata: {result.metadata}")
            
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 