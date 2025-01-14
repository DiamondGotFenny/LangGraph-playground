#this is the ReAct agent that use LangGraph prebuild agent achitecture.
from langchain_core.tools import tool
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

@tool
def get_drinks_menu() -> str:
    """Call this to get the drinks menu."""
    return """
    Drinks Menu:
    - Water
    - Soda
    - Juice
    - Coffee
    - Tea
    """

@tool
def get_food_menu() -> str:
    """Call this to get the food menu."""
    return """
    Food Menu:
    - Burger
    - Pizza
    - Salad
    - Pasta
    - Steak
    """

tools = [get_drinks_menu, get_food_menu]

AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Initialize the AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_4O,
    api_version=AZURE_API_VERSION,
    temperature=0.5,
    max_tokens=3000
)

ReAct_Graph = create_react_agent(llm, tools)

# Initial system message
system_message = ("system", "You are a restaurant waiter. You will greet the user and serve them with restaurant menus, You can only call tools to answer the user's query. After providing the menu items, always ask user if they need anything else, and guide them to ask for menu items.")

# Initialize conversation history
conversation_history = [system_message]

print("Welcome to the restaurant! Type 'q' to quit the conversation.")

while True:
    # Get user input
    user_input = input("\nYou: ")
    
    # Check if user wants to quit
    if user_input.lower() == 'q':
        print("Thank you for visiting! Goodbye!")
        break
    
    # Add user message to conversation history
    conversation_history.append(("user", user_input))
    
    # Prepare inputs for the agent
    inputs = {"messages": conversation_history}
    
    # Get response from the agent
    result = ReAct_Graph.invoke(inputs)
    
     # Print only the last AI message
    ai_messages = [message for message in result['messages'] if isinstance(message, AIMessage)]
    if ai_messages:
        print("\nWaiter:", ai_messages[-1].content)
    
    # Add AI's response to conversation history
    conversation_history.extend([("assistant", message.content) for message in result['messages'] if isinstance(message, AIMessage)])