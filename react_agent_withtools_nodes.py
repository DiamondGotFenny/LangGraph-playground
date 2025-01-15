from langchain_core.tools import tool
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END,START,MessagesState
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
    return """
    Food Menu:
    - Burger
    - Pizza
    - Salad
    - Pasta
    - Steak
    """
# Initialize the AzureChatOpenAI LLM
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_4O,
    api_version=AZURE_API_VERSION,
    temperature=0.5,
    max_tokens=3000
)

tools = [get_drinks_menu, get_food_menu]
tool_node = ToolNode(tools)
model_with_tools=llm.bind_tools(tools)

def should_continue(state:MessagesState):
    messages=state["messages"]
    last_message=messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model_with_tools(state:MessagesState):
    messages=state["messages"]
    response=model_with_tools.invoke(messages)
    return {"messages":[response]}

workflow=StateGraph(MessagesState)

workflow.add_node("agent",call_model_with_tools)
workflow.add_node("tools",tool_node)

workflow.add_edge(START,"agent")
# If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
# # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
# we can use langgraph prebuild 'tools_condition' method, or define by ourself
#workflow.add_conditional_edges("agent",should_continue,["tools",END])
workflow.add_conditional_edges('agent',tools_condition)
workflow.add_edge("tools", "agent")

app=workflow.compile()



# Initial system message
system_message = SystemMessage(content="You are a restaurant waiter. You will greet the user and serve them with restaurant menus, You can only call tools to answer the user's query. After providing the menu items, always ask user if they need anything else, and guide them to ask for menu items.")

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
    conversation_history.append(HumanMessage(content=user_input,name="user"))
    
    # Prepare inputs for the agent
    inputs = {"messages": conversation_history}
    
    # Get response from the agent
    result = app.invoke(inputs)
    
     # Print only the last AI message
    ai_messages = [message for message in result['messages'] if isinstance(message, AIMessage)]
    if ai_messages:
        print("\nWaiter:", ai_messages[-1].content)
    
    # Add AI's response to conversation history
    conversation_history.append(AIMessage(content=ai_messages[-1].content,name="waiter"))
    print("\n chat history", conversation_history)



