import os
from langchain_openai import AzureChatOpenAI
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class GraphState(TypedDict):
    question: Optional[str] = None
    classification: Optional[str] = None
    response: Optional[str] = None

# Load environment variables
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

def classify(question: str) -> str:
    # Initial classification
    classify_prompt = f"""
    As a restaurant waiter, classify the customer's request into one of these categories:
    - "drink": for any beverage orders (water, soda, tea, coffee, etc.)
    - "meal": for any food orders (appetizers, main courses, desserts)
    - "greeting": for general greetings or non-order conversation

    Output ONLY one of these words: "drink", "meal", or "greeting"

    Customer's request: {question}
    Classification:
    """
    
    response = llm.invoke(classify_prompt)
    result = response.content.strip()
    # If response is drink or meal, return directly
    if result.lower() not in ["drink", "meal","greeting"]:
        return "greeting"
    return result.lower()
    

def classify_input_node(state):
    question = state.get('question', '').strip()
    classification = classify(question) 
    return {"classification": classification}

DRINK_MENU = {
    "coffee": ["Espresso", "Americano", "Latte", "Cappuccino"],
    "tea": ["Green Tea", "Black Tea", "Earl Grey", "Chamomile"],
    "soft_drinks": ["Cola", "Sprite", "Orange Juice", "Mineral Water"]
}

FOOD_MENU = {
    "starters": ["Caesar Salad", "Soup of the Day", "Bruschetta"],
    "mains": ["Grilled Salmon", "Beef Steak", "Pasta Carbonara"],
    "desserts": ["Tiramisu", "Ice Cream", "Chocolate Cake"]
}

def handle_drinks_node(state):
    menu_text = "Here are our drink options:\n"
    for category, items in DRINK_MENU.items():
        menu_text += f"\n{category.title()}: {', '.join(items)}"
    return {"response": menu_text}

def handle_meals_node(state):
    menu_text = "Here are our food options:\n"
    for category, items in FOOD_MENU.items():
        menu_text += f"\n{category.title()}: {', '.join(items)}"
    return {"response": menu_text}

def handle_greeting_node(state):
    question = state.get('question').strip()
    greeting_prompt = f"""
        As a friendly waiter, respond to: "{question}"
        Be polite and guide the customer towards our food and drink options.
        Suggest they might want to order something to eat or drink.
        """
    response = llm.invoke(greeting_prompt)
    return {"response": response.content.strip()}

workflow = StateGraph(GraphState)
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_drinks", handle_drinks_node)
workflow.add_node("handle_meals", handle_meals_node)

def decide_next_node(state):
    classification = state.get('classification')
    if classification == "drink":
        return "handle_drinks"
    elif classification == "meal":
        return "handle_meals"
    else:
        return "handle_greeting"

workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_drinks": "handle_drinks",
        "handle_meals": "handle_meals"
    }
)

workflow.set_entry_point("classify_input")
workflow.add_edge('handle_drinks', END)
workflow.add_edge('handle_meals', END)
workflow.add_edge('handle_greeting', END)

app = workflow.compile()

# Loop to accept user input
while True:
    user_input = input("Enter your question (or 'q' to quit): ").strip()
    
    # Exit the loop if the user types 'q'
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    
    # Invoke the workflow with the user's input
    inputs = {"question": user_input}
    result = app.invoke(inputs)
    
    # Print the response
    print("Response:", result.get("response", "No response generated."))
    print()  # Add a blank line for readability