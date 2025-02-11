"""
This script demonstrates an LLM-powered restaurant waiter with a custom LangGraph state schema for order tracking:
(Refactored version with explicit input and output schemas for StateGraph)
1. Polite conversation and menu Q&A.
2. Order creation with unique order ID and status tracking using custom state.
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
import uuid
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Dict, Union, Optional, Literal, TypedDict
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AnyMessage, trim_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, START, add_messages
from logger_config import setup_logger
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv(find_dotenv())

def ensure_log_file(log_file_path: str) -> str:
    """Ensure log file exists, create if it doesn't, and return the path."""
    log_path = Path(log_file_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        return str(log_path)
    except Exception as e:
        print(f"Warning: Could not create/access log file: {e}")
        return "waiter_react_agent.log"  # Fallback to current directory

log_file_path = ensure_log_file("waiter_react_agent.log")
logger = setup_logger(log_file_path)

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
    "Red Wine": 9.00, "White Wine": 9.00, "Cocktail": 12.00,
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
    "Bar": {"Red Wine": 30, "White Wine": 30, "Cocktail": 25, "Beer": 40},
    "Coffee/Tea Bar": {"Espresso": 50, "Cappuccino": 40, "Latte": 45, "Green Tea": 30, "Black Tea": 35},
}
# Global inventory
DEPARTMENT_STOCKS = {dept: dict(items) for dept, items in initial_stocks.items()}

# -------------------------- Helper Functions -------------------------- #
def get_new_order_id() -> str:
    """Generate a new unique order id using UUID."""
    return str(uuid.uuid4())

def check_and_update_stock(item_name: str, quantity: int) -> str:
    """
    Check if the specified item is in stock in the relevant department.
    If in stock, decrement the inventory and return 'fulfilled'.
    If not enough stock, return 'insufficient_stock'.
    If item not found, return 'item_not_found'.
    """
    logger.info(f"Checking stock for item: {item_name}, quantity: {quantity}")
    for department, items in DEPARTMENT_STOCKS.items():
        if item_name in items:
            if items[item_name] >= quantity:
                items[item_name] -= quantity
                logger.info(f"Stock fulfilled: {item_name} - {quantity} from {department}")
                return f"fulfilled in {department}"
            else:
                logger.warning(f"Insufficient stock for {item_name} in {department}")
                return f"insufficient_stock in {department}"
    logger.error(f"Item not found in any department: {item_name}")
    return "item_not_found"

def should_summarize(state: "RestaurantOrderState") -> bool:
    logger.info(f"Conversation rounds: {state['conversation_rounds']}")
    return state["conversation_rounds"] >= 6

def conversation_summarizer(state: "RestaurantOrderState") -> "RestaurantOrderState":
    """Generate and store conversation summary, then update the state."""
    global system_message
    logger.info("Generating conversation summary")
    messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    logger.info(f"Total messages in conversation_summarizer: {len(messages)}")
    summary_prompt = [
        SystemMessage(content="Summarize the conversation provided below. Focus on extracting key details such as ordered items, quantities, and payment status. Only summarize information explicitly stated."),
        HumanMessage(content="\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Waiter'}: {m.content}"
            for m in messages[:12]
        ]))
    ]

    summary = gemini_llm.invoke(summary_prompt).content
    logger.info(f"Generated summary: {summary}")
    # Remove non-user/non-ai messages except the system message and keep last 4 messages
    new_messages = [system_message] + messages[-4:]
    if state['summary']:
        summary = f"{state['summary']}\n{summary}"
    updated_state_dict = {
        "messages": new_messages,
        "order_id": state["order_id"],
        "order_items": state["order_items"],
        "order_status": state["order_status"],
        "total_cost": state["total_cost"],
        "payment_status": state["payment_status"],
        "conversation_rounds": 0,  # Reset counter
        "summary": summary
    }
    updated_state = RestaurantOrderState.validate_state(updated_state_dict)
    logger.info(f"State before summarizer returns: {updated_state}")
    return updated_state

# -------------------------- Pydantic Models for State -------------------------- #
class OrderItem(BaseModel):
    """Represents a single item in an order."""
    name: str
    quantity: int = Field(gt=0)
    department: str = ""
    status: Literal["pending", "fulfilled"] = "pending"

    @field_validator('name')
    @classmethod
    def validate_menu_item(cls, value):
        if value not in menu_prices:
            raise ValueError(f"Item '{value}' is not in the menu")
        return value

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Quantity must be a positive integer")
        return value

class RestaurantOrderStateModel(BaseModel):
    """Pydantic model for restaurant order state."""
    messages: Annotated[List[AnyMessage], add_messages]
    order_id: Optional[str] = Field(default="", description="Unique identifier for the order")
    order_items: List[OrderItem] = Field(default_factory=list, description="List of items in the order")
    order_status: Literal["pending", "processing", "fulfilled", "billed", "paid", "partially_fulfilled"] = Field(
        default="pending",
        description="Current status of the order"
    )
    total_cost: float = Field(default=0.0, ge=0, description="Total cost of the order")
    payment_status: Literal["pending", "paid", "failed"] = Field(
        default="pending",
        description="Status of payment"
    )
    conversation_rounds: int = Field(default=0, description="Number of conversation exchanges")
    summary: str = Field(default="", description="Condensed conversation summary")

    @field_validator('total_cost')
    @classmethod
    def validate_total_cost(cls, value):
        if value < 0:
            raise ValueError("Total cost cannot be negative")
        return value

    @field_validator('order_status')
    @classmethod
    def validate_order_status(cls, value):
        valid_statuses = ["pending", "processing", "fulfilled", "billed", "paid", "partially_fulfilled"]
        if value not in valid_statuses:
            raise ValueError(f"Invalid order status. Must be one of: {', '.join(valid_statuses)}")
        return value

    @field_validator('payment_status')
    @classmethod
    def validate_payment_status(cls, value):
        valid_statuses = ["pending", "paid", "failed"]
        if value not in valid_statuses:
            raise ValueError(f"Invalid payment status. Must be one of: {', '.join(valid_statuses)}")
        return value

    class Config:
        arbitrary_types_allowed = True

class RestaurantOrderState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    order_id: Optional[str]
    order_items: List[OrderItem]
    order_status: str
    total_cost: float
    payment_status: str
    conversation_rounds: int
    summary: str

    @classmethod
    def validate_state(cls, state_dict: dict) -> 'RestaurantOrderState':
        try:
            if "order_items" in state_dict:
                state_dict["order_items"] = [
                    OrderItem(**item) if isinstance(item, dict) else item
                    for item in state_dict["order_items"]
                ]
            validated = RestaurantOrderStateModel(**state_dict)
            return cls(
                messages=validated.messages,
                order_id=validated.order_id,
                order_items=[item.model_dump() for item in validated.order_items],
                order_status=validated.order_status,
                total_cost=validated.total_cost,
                payment_status=validated.payment_status,
                conversation_rounds=validated.conversation_rounds,
                summary=validated.summary
            )
        except ValidationError as e:
            logger.error(f"State validation error: {str(e)}")
            raise

class InputState(TypedDict):
    messages: List[AnyMessage]
    conversation_rounds: int

class OutputState(TypedDict):
    messages: List[AIMessage]

# -------------------------- Tools Section -------------------------- #
from langchain_core.messages import BaseMessage

@tool
def get_drinks_menu() -> str:
    """Call this to get the drinks menu."""
    return """
    Drinks Menu:
    - Red Wine
    - White Wine
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
def get_menu_item_price(item_name: str) -> str:
    """Call this to get the price of a specific menu item."""
    if item_name in menu_prices:
        return f"{item_name} costs ${menu_prices[item_name]:.2f}"
    return f"Item '{item_name}' not found in menu"

@tool
def get_restaurant_info() -> str:
    """Call this to get information about the restaurant."""
    return """
    === VILLA TOSCANA ===

    🏰 About Us:
    Established in 1985, Villa Toscana sits in a restored 19th-century mansion in the heart of the city.
    Our restaurant brings authentic Tuscan flavors with a modern twist to your table.

    👨‍🍳 Our Team:
    - Owner: Marco Rossi (3rd generation restaurateur)
    - Executive Chef: Isabella Chen
        - Former sous chef at 3-Michelin-starred Le Bernardin
        - James Beard Rising Star Chef 2022
    - Sommelier: James Thompson (Court of Master Sommeliers certified)

    🏆 Awards & Recognition:
    - Michelin Star (2020-2024)
    - Wine Spectator Award of Excellence (2018-2024)
    - Best Italian Restaurant - City Dining Awards 2023
    - "Top 50 Restaurants in America" - Bon Appétit Magazine 2022

    ⏰ Hours of Operation:
    - Lunch: Tuesday-Sunday, 11:30 AM - 2:30 PM
    - Dinner: Tuesday-Sunday, 5:30 PM - 10:00 PM
    - Closed on Mondays

    🎉 Special Events:
    - Weekly wine tasting events (Thursday evenings)
    - Monthly cooking classes with Chef Isabella
    - Private dining rooms available for special occasions

    For reservations: +1 (555) 123-4567
    Address: 123 Olive Garden Street, Metropolis, MB 12345
    """

@tool
def create_order(order_items: List[OrderItem], current_order_id: Optional[str] = None) -> str:
    """
    Creates or updates the current order.
    If no current_order_id is provided, a new UUID is generated.
    The provided order_items (a list of OrderItem objects) are used for the order.
    """
    if current_order_id and current_order_id.strip() != "":
        order_id = current_order_id
        action = "updated"
    else:
        order_id = get_new_order_id()
        action = "created"
    subtotal = sum(menu_prices.get(item.name, 0.0) * item.quantity for item in order_items)
    tip = round(subtotal * 0.15, 2)
    total = round(subtotal + tip, 2)
    return f"Order {order_id} {action} successfully with status 'pending'. Total cost (including 15% tip) is ${total:.2f}."

@tool
def process_order(order_id: str, order_items: List[OrderItem]) -> str:
    """
    Processes the current order.
    Checks stock for each item and updates the status accordingly.
    """
    if not order_id:
        return "No order found. Please create an order first."
    all_fulfilled = True
    for item in order_items:
        if item.status == "pending":
            result = check_and_update_stock(item.name, item.quantity)
            if "fulfilled" in result:
                dept = result.split(" in ")[-1]
                item.department = dept
                item.status = "fulfilled"
            elif "insufficient_stock" in result:
                all_fulfilled = False
            else:
                all_fulfilled = False
    status_message = "fully fulfilled" if all_fulfilled else "partially fulfilled"
    return f"Order {order_id} processed and is {status_message}."

@tool
def cashier_calculate_total(order_id: str, order_items: List[OrderItem]) -> float:
    """
    Calculates the total for the order (items’ prices plus a 15% tip).
    """
    subtotal = sum(menu_prices.get(item.name, 0.0) * item.quantity for item in order_items)
    tip = round(subtotal * 0.15, 2)
    total = round(subtotal + tip, 2)
    logger.info(f"Calculated total for order {order_id}: ${total}")
    return total

@tool
def check_payment(amount: float, method: str, total_due: float, order_id: str) -> str:
    """
    Processes the payment for the order.
    For cash, checks for sufficient funds; 
    for card, uses an 80% chance of success.
    """
    logger.info(f"Processing payment: ${amount} via {method} for order {order_id}")
    if method.lower() == "cash":
        if amount < total_due:
            logger.warning(f"Insufficient cash: ${amount} < ${total_due}")
            return "insufficient_funds"
        change = round(amount - total_due, 2)
        return f"cash_ok with change {change}"
    elif method.lower() == "card":
        if random.random() < 0.8:
            logger.info(f"Card payment successful for order {order_id}")
            return "valid"
        else:
            logger.warning(f"Card payment failed for order {order_id}")
            return "invalid"
    else:
        logger.error(f"Unknown payment method: {method}")
        return "unknown_method"

# -------------------------- LLM and Workflow Initialization -------------------------- #
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
OPENAI_MODEL_O1MINI = os.getenv("OPENAI_MODEL_O1MINI")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_OPENAI_4OMINI = os.getenv("OPENAI_MODEL_4OMINI")

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_4O,
    api_version=AZURE_API_VERSION,
    temperature=0.5,
    max_tokens=4000
)

DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
deepseek_llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_ENDPOINT,
    model=DEEPSEEK_MODEL,
    temperature=0.5,
    max_tokens=4096
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_FLASH20 = os.getenv("GEMINI_MODEL_FLASH20")
GEMINI_MODEL_FLASH15 = os.getenv("GEMINI_MODEL_FLASH15")
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_FLASH20,
    api_key=GEMINI_API_KEY,
    temperature=0.5,
    max_tokens=4096,
    timeout=60,
    transport="rest"
)

tools = [
    get_drinks_menu,
    get_food_menu,
    create_order,
    process_order,
    cashier_calculate_total,
    check_payment,
    get_restaurant_info,
    get_menu_item_price,
]
tool_node = ToolNode(tools)
model_with_tools = gemini_llm.bind_tools(tools)

def should_continue(state: RestaurantOrderState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    logger.info(f"Last tool message: {last_message}")
    if last_message.tool_calls:
        logger.info("Last message is a tool call")
        return "tools"
    logger.info("Last message is not a tool call")
    return END

def call_model_with_tools(state: RestaurantOrderState) -> RestaurantOrderState:
    messages = state['messages']
    logger.info(f"State in call_model_with_tools: {state}")
    summary = state.get('summary', '')
    global system_message
    if state["conversation_rounds"] >= 6:
        state = conversation_summarizer(state)
        messages = state['messages']
        filtered_messages = [
            msg for msg in messages 
            if not (isinstance(msg, AIMessage) and msg.content == '' and (msg.additional_kwargs.get('function_call') or msg.tool_calls))
        ]
        other_messages = [msg for msg in filtered_messages if not isinstance(msg, SystemMessage)]
        if summary:
            summary_message = AIMessage(content=f"Conversation Summary:\n{summary}")
            messages = [system_message, summary_message] + other_messages
        else:
            messages = filtered_messages
        app.update_state(values={"messages": messages}, config=config)
        logger.info(f"Updated state after summarizer: {state}")
    try:
        if isinstance(messages[-1], HumanMessage) and summary:
            summary_message = AIMessage(content=f"Conversation Summary:\n{state['summary']}")
            messages = [system_message, summary_message] + messages[-4:]
        logger.info(f"LLM input: {messages}")
        messages = [msg for msg in messages if not (isinstance(msg, AIMessage) and msg.content == '' and not msg.additional_kwargs.get('function_call'))]
        response = model_with_tools.invoke(messages)
        logger.info(f"LLM response: {response}")
        updated_state_dict = {
            "messages": [response],
            "order_id": state.get('order_id', ""),
            "order_items": state.get('order_items', []),
            "order_status": state.get('order_status', 'pending'),
            "total_cost": state.get('total_cost', 0.0),
            "payment_status": state.get('payment_status', 'pending'),
            "conversation_rounds": state.get('conversation_rounds', 0),
            "summary": state.get('summary', '')
        }
        updated_state = RestaurantOrderState.validate_state(updated_state_dict)
        logger.info(f"Updated state: {updated_state}")
        return updated_state
    except Exception as e:
        logger.error(f"LLM invoke error: {str(e)}", exc_info=True)
        error_message = AIMessage(content="I apologize, but I'm having trouble processing your request. Could you please try again?")
        updated_state_dict = {
            "messages": [error_message],
            "order_id": state.get('order_id', ""),
            "order_items": state.get('order_items', []),
            "order_status": state.get('order_status', 'pending'),
            "total_cost": state.get('total_cost', 0.0),
            "payment_status": state.get('payment_status', 'pending'),
            "conversation_rounds": state.get('conversation_rounds', 0),
            "summary": state.get('summary', '')
        }
        updated_state = RestaurantOrderState.validate_state(updated_state_dict)
        return updated_state

# Build the state graph
workflow = StateGraph(RestaurantOrderState, input=InputState, output=OutputState)
memory = MemorySaver()
workflow.add_node("agent", call_model_with_tools)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Updated system prompt now includes the state schema
system_message = SystemMessage(
    content=(
       "You are a waiter at Villa Toscana, an upscale Italian restaurant. "
       "Current Order State Schema:\n"
       "  - order_id: Unique order identifier (UUID string)\n"
       "  - order_items: List of items (each with name, quantity, department, and status: pending/fulfilled)\n"
       "  - order_status: One of pending, processing, fulfilled, billed, paid, or partially_fulfilled\n"
       "  - total_cost: Total cost of the order\n"
       "  - payment_status: One of pending, paid, or failed\n"
       "  - conversation_rounds: Number of dialogue exchanges\n"
       "  - summary: A short summary of the conversation\n\n"
       "You will greet the user and help them with restaurant menus, orders, billing, and payment processing. "
       "Always use create_order to create/update a customer's order. If an order already exists in the state, update its order_items. "
       "When providing answers or taking actions, consult and update the state accordingly. "
       "Only call tools to answer the user's query. Remain polite, confirm orders, and handle small talk briefly before refocusing on restaurant matters."
    )
)

if __name__ == "__main__":
    logger.info("Starting restaurant waiter service with custom state")
    print("Welcome to the restaurant! Type 'q' to quit the conversation.")
    try:
        app.update_state(values={
            "messages": [system_message],
            "order_id": "",
            "order_items": [],
            "order_status": "pending",
            "total_cost": 0.0,
            "payment_status": "pending",
            "conversation_rounds": 0,
            "summary": ""
        }, config=config)
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'q':
                logger.info("User ended conversation")
                print("Thank you for visiting! Goodbye!")
                break
            logger.info(f"User input: {user_input}")
            user_message = HumanMessage(content=user_input, name="user")
            current_state = app.get_state(config)
            app.update_state(values={"conversation_rounds": current_state[0]['conversation_rounds'] + 1}, config=config)
            try:
                updated_state_dict = {
                    "messages": list(current_state[0]['messages']) + [user_message]
                }
                validated_state = RestaurantOrderState.validate_state(updated_state_dict)
                result = app.invoke(updated_state_dict, config=config)
                ai_messages = [msg for msg in result['messages'] if isinstance(msg, AIMessage)]
                if ai_messages:
                    print("\nWaiter:", ai_messages[-1].content)
                    logger.info(f"AI response: {ai_messages[-1].content}")
            except ValidationError as e:
                logger.error(f"State validation error: {str(e)}")
                print("\nWaiter: I apologize, but there was an error with your order. Please try again.")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        print("\nWaiter: I apologize, but our system is currently unavailable. Please try again later.")