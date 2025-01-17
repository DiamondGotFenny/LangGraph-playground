
"""
waiter_react_agent.py

This script demonstrates an LLM-powered restaurant waiter:
1. Polite conversation and menu Q&A.
2. Order creation with unique order ID and status tracking.
3. Communication with multiple departments for stock checks and order fulfillment.
4. Billing and payment (with a 15% tip, random 20% chance of invalid credit card).
5. Graceful error handling and request for alternate payment method if necessary.

Requirements:
- python-dotenv for reading environment variables from a .env file (optional).
- The "langchain_core" and "langgraph" modules in your environment, 
  or adapt the code to your own tool/chain management libraries.
"""

import os
import random
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Union
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, START, MessagesState

_ = load_dotenv(find_dotenv()) 

# -------------------------- Menus and Pricing -------------------------- #
menu_prices = {
    "Bruschetta": 8.50, "Caprese Salad": 9.00, "Shrimp Cocktail": 12.50,
    "Salmon Fillet": 18.00, "Chicken Breast": 15.00, "Vegetable Stir-Fry": 14.00,
    "Filet Mignon": 35.00, "Lobster Tail": 40.00, "Rack of Lamb": 32.00,
    "Mashed Potatoes": 5.00, "Grilled Asparagus": 6.00, "Roasted Vegetables": 5.50,
    "Chocolate Cake": 8.00, "Cheesecake": 7.50, "Tiramisu": 9.00,
    "Spaghetti Carbonara": 16.00, "Fettuccine Alfredo": 17.00, "Lasagna": 19.00,
    "Ribeye Steak": 30.00, "BBQ Ribs": 25.00, "Grilled Salmon": 28.00,
    "Caesar Salad": 10.00, "Greek Salad": 9.50, "Fruit Platter": 12.00,
    "Red Wine (glass)": 9.00, "White Wine (glass)": 9.00, "Cocktail": 12.00,
    "Beer": 7.00, "Espresso": 3.00, "Cappuccino": 4.50, "Latte": 4.50,
    "Green Tea": 3.50, "Black Tea": 3.50,
}

# Initial department inventories
initial_stocks = {
    "Appetizer Station": {"Bruschetta": 10, "Caprese Salad": 15, "Shrimp Cocktail": 20},
    "Entrée Station": {"Salmon Fillet": 15, "Chicken Breast": 20, "Vegetable Stir-Fry": 10},
    "Main Course Station": {"Filet Mignon": 10, "Lobster Tail": 8, "Rack of Lamb": 12},
    "Side Dish Station": {"Mashed Potatoes": 25, "Grilled Asparagus": 30, "Roasted Vegetables": 20},
    "Pastry/Dessert Station": {"Chocolate Cake": 5, "Cheesecake": 7, "Tiramisu": 10},
    "Pasta Station": {"Spaghetti Carbonara": 15, "Fettuccine Alfredo": 18, "Lasagna": 12},
    "Grill Station": {"Ribeye Steak": 12, "BBQ Ribs": 15, "Grilled Salmon": 18},
    "Cold Station": {"Caesar Salad": 20, "Greek Salad": 15, "Fruit Platter": 10},
    "Bar": {"Red Wine (glass)": 30, "White Wine (glass)": 30, "Cocktail": 25, "Beer": 40},
    "Coffee/Tea Bar": {"Espresso": 50, "Cappuccino": 40, "Latte": 45, "Green Tea": 30, "Black Tea": 35},
}

# Global data structures
DEPARTMENT_STOCKS = {dept: dict(items) for dept, items in initial_stocks.items()}

# Example: store orders in a global dictionary
# orders[order_id] = {
#    "items": [{"name": <str>, "quantity": <int>, "department": <str>, "status": <str>}],
#    "status": "pending" | "partially_fulfilled" | "fulfilled" | "billed" | "paid",
#    "total_cost": 0,
#    "paid_amount": 0,
#    ...
# }
orders = {}
order_counter = 0


def get_new_order_id() -> int:
    """
    Returns a new unique order ID.
    """
    global order_counter
    order_counter += 1
    return order_counter


# -------------------------- Tools Section -------------------------- #
@tool
def get_drinks_menu() -> str:
    """Call this to get the drinks menu."""
    return """
    Drinks Menu:
    - Red Wine (glass)
    - White Wine (glass)
    - Cocktail
    - Beer
    - Espresso
    - Cappuccino
    - Latte
    - Green Tea
    - Black Tea
    """


@tool
def get_food_menu() -> str:
    """Call this to get the food menu."""
    return """
    Food Menu:
    (Appetizers)
    - Bruschetta
    - Caprese Salad
    - Shrimp Cocktail
    
    (Entrées)
    - Salmon Fillet
    - Chicken Breast
    - Vegetable Stir-Fry
    
    (Main Courses)
    - Filet Mignon
    - Lobster Tail
    - Rack of Lamb
    
    (Side Dishes)
    - Mashed Potatoes
    - Grilled Asparagus
    - Roasted Vegetables
    
    (Pasta)
    - Spaghetti Carbonara
    - Fettuccine Alfredo
    - Lasagna

    (Grill Station)
    - Ribeye Steak
    - BBQ Ribs
    - Grilled Salmon

    (Cold Station)
    - Caesar Salad
    - Greek Salad
    - Fruit Platter

    (Desserts)
    - Chocolate Cake
    - Cheesecake
    - Tiramisu
    """


@tool
def check_and_update_stock(item_name: str, quantity: int) -> str:
    """
    Check if the specified item is in stock in the relevant department.
    If in stock, decrement the inventory and return 'fulfilled'.
    If not enough stock, return 'insufficient_stock'.
    If item not found, return 'item_not_found'.
    """
    # 1) Find which department has this item
    for department, items in DEPARTMENT_STOCKS.items():
        if item_name in items:
            # 2) Check if we have enough stock
            if items[item_name] >= quantity:
                # reduce stock
                items[item_name] -= quantity
                return f"fulfilled in {department}"
            else:
                return f"insufficient_stock in {department}"
    return "item_not_found"


@tool
def cashier_calculate_total(order_id: int) -> float:
    """
    Calculates the total cost of the entire order (sum of item prices)
    plus 15% tip. Updates the 'total_cost' field in the order.
    Returns the total amount.
    """
    if order_id not in orders:
        return 0.0

    order_data = orders[order_id]
    subtotal = 0.0
    for item in order_data["items"]:
        name = item["name"]
        qty = item["quantity"]
        price = menu_prices.get(name, 0.0)
        subtotal += price * qty

    tip_amount = subtotal * 0.15
    total = round(subtotal + tip_amount, 2)

    # Update order record
    order_data["total_cost"] = total
    return total


@tool
def check_payment(amount: float, method: str, order_id: int = 0) -> str:
    """
    Processes the payment. 
    - method = "cash": returns "cash_ok" or "insufficient_funds" if amount < total
    - method = "card": 80% chance "valid", 20% chance "invalid"
    If payment is successful, updates order status to "paid".
    """
    if order_id not in orders:
        return "order_not_found"

    order_data = orders[order_id]
    total_due = order_data["total_cost"]

    if method.lower() == "cash":
        if amount < total_due:
            return "insufficient_funds"
        # Payment success; calculate change
        change = round(amount - total_due, 2)
        order_data["status"] = "paid"
        return f"cash_ok with change {change}"

    elif method.lower() == "card":
        # 80% success
        if random.random() < 0.8:
            order_data["status"] = "paid"
            return "valid"
        else:
            return "invalid"

    else:
        return "unknown_method"


@tool
def create_order(order_items: List[Dict[str, Union[str, int]]]) -> str:
    """
    Creates a new order with status 'pending', given a list of order items in JSON format, 
    for example:
    [
        {"name": "Bruschetta", "quantity": 2},
        {"name": "Lobster Tail", "quantity": 1}
    ]
    Returns the newly created order ID.
    """
    order_id = get_new_order_id()

    # Convert the incoming list of dicts into our expected "items" structure
    parsed_items = []
    for item_info in order_items:
        item_name = item_info.get("name", "").strip()
        qty = item_info.get("quantity", 1)
        parsed_items.append({
            "name": item_name, 
            "quantity": qty,
            "department": "", 
            "status": "pending"
        })
    
    orders[order_id] = {
        "items": parsed_items,
        "status": "pending",
        "total_cost": 0.0,
    }
    return f"New order {order_id} created with status 'pending'."


@tool
def process_order(order_id: int) -> str:
    """
    Processes an existing order:
    1. Checks each item's department stock via check_and_update_stock.
    2. If insufficient stock for any item, that item remains pending, 
       and the overall order might be partially_fulfilled.
    3. If all items fulfilled, order is marked 'fulfilled'.
    """
    if order_id not in orders:
        return "order_not_found"
    
    order_data = orders[order_id]
    if order_data["status"] not in ["pending", "partially_fulfilled"]:
        return f"cannot_process_order_in_state_{order_data['status']}"

    all_fulfilled = True
    for item in order_data["items"]:
        if item["status"] == "pending":
            # call tool check_and_update_stock item_name, quantity
            result = check_and_update_stock(item["name"], item["quantity"])
            if "fulfilled" in result:
                # Mark item fulfilled; record department
                dept = result.split(" in ")[-1]
                item["department"] = dept
                item["status"] = "fulfilled"
            elif "insufficient_stock" in result:
                all_fulfilled = False
            else:
                # item not found or another error
                all_fulfilled = False

    if all_fulfilled:
        order_data["status"] = "fulfilled"
        return f"Order {order_id} is fully fulfilled."
    else:
        # If at least one item remains "pending", the order is partially fulfilled
        order_data["status"] = "partially_fulfilled"
        return f"Order {order_id} is partially fulfilled. Some items are not available or still pending."


# -------------------------- LLM and Workflow Initialization -------------------------- #

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
    max_tokens=4000
)

# Add our expanded tools
tools = [
    get_drinks_menu, 
    get_food_menu, 
    create_order,
    process_order,
    cashier_calculate_total,
    check_payment
]

tool_node = ToolNode(tools)
model_with_tools = llm.bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model_with_tools(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages":[response]}

# Build the state graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model_with_tools)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
# If the latest message from assistant is a tool call -> tools_condition routes to tools
# If the latest message from assistant is not a tool call -> route to END
workflow.add_conditional_edges('agent', tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Initial system message
system_message = SystemMessage(
    content=(
        "You are a restaurant waiter. You will greet the user and serve them with restaurant menus, "
        "manage orders, check availability, handle billing and payments, and communicate with the "
        "virtual restaurant departments to fulfill orders. "
        "You can only call tools to answer the user's query or perform operations. "
        "Always remain polite, confirm orders, provide recommendations, and handle small talk briefly "
        "before steering back to restaurant matters. After providing information or fulfilling requests, "
        "ask the user if they need anything else."
        "When the user wants to order items, call create_order with a JSON list of objects. "
    "For each item, provide {\\\"name\\\": <str>, \\\"quantity\\\": <int>}. For instance: "
    "create_order({\\\"order_items\\\": ["
    "{\\\"name\\\": \\\"Bruschetta\\\", \\\"quantity\\\": 2}, "
    "{\\\"name\\\": \\\"Lobster Tail\\\", \\\"quantity\\\": 1}"
    "]})."
    "Remember only produce valid JSON."
    )
)

# Initialize conversation history
conversation_history = [system_message]

if __name__ == "__main__":
    print("Welcome to the restaurant! Type 'q' to quit the conversation.")

    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to quit
        if user_input.lower() == 'q':
            print("Thank you for visiting! Goodbye!")
            break
        
        # Add user message to conversation history
        conversation_history.append(HumanMessage(content=user_input, name="user"))
        
        # Prepare inputs for the agent
        inputs = {"messages": conversation_history}
        
        # Get response from the agent
        result = app.invoke(inputs)
        
        # Print only the last AI message
        ai_messages = [message for message in result['messages'] if isinstance(message, AIMessage)]
        if ai_messages:
            print("\nWaiter:", ai_messages[-1].content)
        
        # Add AI's response to conversation history
        conversation_history.append(AIMessage(content=ai_messages[-1].content, name="waiter"))
        
        # Debug (optional): see conversation history
        # print("\n chat history", conversation_history)