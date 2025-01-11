from langchain_core.tools import tool
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END,MessagesState
from langgraph.prebuilt import tools_condition
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
    print("Food Menu:")
    return """
    Food Menu:
    - Burger
    - Pizza
    - Salad
    - Pasta
    - Steak
    """


tools = [get_drinks_menu, get_food_menu]
tool_node = ToolNode(tools)

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
model_with_tools=llm.bind_tools(tools)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a restaurant waiter. You will greet the user and serve them with restaurant menus, You can only call tools to answer the user's query. After providing the menu items, always ask user if they need anything else, and guide them to ask for menu items. ",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
result=tool_node.invoke({"messages": [model_with_tools.invoke("give me the food menu")]})
print(result)



