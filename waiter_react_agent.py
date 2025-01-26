####################
"""
waiter_react_agent.py

This script demonstrates an LLM-powered restaurant waiter with a custom LangGraph state schema for order tracking:
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
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Dict, Union, Optional, Literal,TypedDict
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage,AnyMessage,RemoveMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, START,add_messages
from logger_config import setup_logger
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv(find_dotenv())


def ensure_log_file(log_file_path: str) -> str:
    """Ensure log file exists, create if it doesn't, and return the path."""
    log_path = Path(log_file_path)
    try:
        # Create parent directories if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        log_path.touch(exist_ok=True)
        return str(log_path)
    except Exception as e:
        print(f"Warning: Could not create/access log file: {e}")
        return "waiter_react_agent.log"  # Fallback to current directory

_ = load_dotenv(find_dotenv())

# Setup logger with guaranteed log file
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
system_message = SystemMessage(
    content=(
       "You are a waiter at Villa Toscana, an upscale Italian restaurant. You will greet the user and serve them "
        "with restaurant menus, manage orders, check availability, handle billing and payments, and communicate with "
        "the virtual restaurant departments to fulfill orders. You can provide information about our restaurant's "
        "history, team, and accolades when asked. "
        "You can only call tools to answer the user's query or perform operations. "
        "Always remain polite, confirm orders, provide recommendations, and handle small talk briefly "
        "before steering back to restaurant matters. After providing information or fulfilling requests, "
        "ask the user if they need anything else."
        "Always use create_order first before processing an order."
        "You only need to create one order for all items, then process the order."
        "When the user wants to order items, call create_order with a JSON list of objects. "
        "For each item, provide {\\\"name\\\": <str>, \\\"quantity\\\": <int>}. For instance: "
        "create_order({\\\"order_items\\\": ["
        "{\\\"name\\\": \\\"Bruschetta\\\", \\\"quantity\\\": 2}, "
        "{\\\"name\\\": \\\"Lobster Tail\\\", \\\"quantity\\\": 1}"
        "]})."
        "Payment Protocol: Only initiate billing when the user explicitly: \n"
        "1. States they're finished (e.g., 'I'm done', 'That's all') \n"
        "2. Directly requests the bill (e.g., 'Check please', 'Can we pay?') \n"
        "3. Asks about payment (e.g., 'How much do I owe?') \n"
        "Never suggest payment first - always wait for customer initiation. "
        "If order modifications continue after billing request, recalculate totals."
        "Before create_order, always verify that the customer's requested items exactly match the official menu names "
        " Convert colloquial terms to standardized menu names "
        "(e.g., 'steak' → 'Ribeye Steak', 'iced tea' → 'Black Tea'). If ambiguous, politely clarify with the customer. "
        "Incorrect names will fail inventory checks and delay order processing."
        "To not call these tools unnecessarily in other situations."
        "When the user asks about the price of a specific item, "
        "use the get_menu_item_price tool to provide accurate pricing information. "
        "Always verify the exact menu name before checking prices."
        "To use 'get_food_menu' and 'get_drinks_menu' only when the user explicitly asks for the menu(e.g., \"show me the menu,\" \"can I see the menu?\") or verify the exact menu item name, if there is no full menu in the conversation history."
    )
)

def get_new_order_id() -> int:
    """
    Returns a new unique order ID.
    """
    global order_counter
    order_counter += 1
    return order_counter


# -------------------------- LangGraph State Definition -------------------------- #
class OrderItem(BaseModel):
    """Represents a single item in an order"""
    name: str
    quantity: int = Field(gt=0)  # Ensure quantity is greater than 0
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
    """Pydantic model for restaurant order state"""
    messages: Annotated[List[AnyMessage], add_messages]
    order_id: Optional[int] = Field(default=0, description="Unique identifier for the order")
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
    order_id: Optional[int]
    order_items: List[OrderItem]
    order_status: str
    total_cost: float
    payment_status: str
    conversation_rounds: int
    summary: str

    @classmethod
    def validate_state(cls, state_dict: dict) -> 'RestaurantOrderState':
        """Validate state using Pydantic model"""
        try:
            # Convert order_items to proper format for Pydantic validation
            if "order_items" in state_dict:
                state_dict["order_items"] = [
                    OrderItem(**item) if isinstance(item, dict) else item
                    for item in state_dict["order_items"]
                ]

            # Validate using Pydantic model
            validated = RestaurantOrderStateModel(**state_dict)

            # Convert back to TypedDict format
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
    """Call this to get the price of a specific menu item.
    Returns the price if item exists, or an error message if not found."""
    if item_name in menu_prices:
        return f"{item_name} costs ${menu_prices[item_name]:.2f}"
    return f"Item '{item_name}' not found in menu"

@tool
def get_restaurant_info() -> str:
    """Call this to get information about the restaurant, including its history, hours, and accolades."""
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

def should_summarize(state: RestaurantOrderState) -> bool:
    logger.info(f"Conversation rounds: {state['conversation_rounds']}")
    return state["conversation_rounds"] >= 6



def conversation_summarizer(state: RestaurantOrderState) -> dict:
    """Generate and store conversation summary"""
    global system_message
    logger.info("Generating conversation summary")
    messages = [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    logger.info(f"Total messages in conversation_summarizer: {len(messages)}")
    # Create summary prompt
    summary_prompt = [
        SystemMessage(content="Condense this conversation while preserving order details, dietary preferences, and payment status. Keep important customers information and preferences."),
        HumanMessage(content="\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Waiter'}: {m.content}"
            for m in messages[-16:]  # Last 8 rounds (2 messages per round)
        ]))
    ]

    # Generate summary
    summary = deepseek_llm.invoke(summary_prompt).content
    logger.info(f"Generated summary: {summary}")
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]
    logger.info(f"Deleting messages:    {delete_messages}")
    logger.info(f"length of state messages before deletion: {len(state['messages'])}")
    logger.info(f"Messages before summarization: {state['messages']}") # Log messages before summarization

    new_messages = [system_message] + messages[-6:] 
    logger.info(f"Messages after summarization and deletion: {new_messages}") # Log messages after summarization

    # Update state
    updated_state = {
        "messages": new_messages,
        "order_id": state["order_id"],
        "order_items": state["order_items"],
        "order_status": state["order_status"],
        "total_cost": state["total_cost"],
        "payment_status": state["payment_status"],
        "conversation_rounds": 0,  # Reset counter
        "summary": f"Previous Summary: {state['summary']}\nNew Summary: {summary}"
    }
    logger.info(f"State before summarizer returns: {updated_state}") 
    return updated_state


@tool
def cashier_calculate_total(order_id: int) -> float:
    """
    Calculates the total cost of the entire order (sum of item prices)
    plus 15% tip. Updates the 'total_cost' field in the order.
    Returns the total amount.
    """
    logger.info(f"Calculating total for order {order_id}")
    if not orders:
        logger.warning("No orders exist in the system")
        return "create an order first"

    if order_id not in orders:
        logger.error(f"Order {order_id} not found")
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


    logger.info(f"Order {order_id} total calculated: ${total}")
    return total


@tool
def check_payment(amount: float, method: str, order_id: int ) -> str:
    """
    Processes the payment.
    - method = "cash": returns "cash_ok" or "insufficient_funds" if amount < total
    - method = "card": 80% chance "valid", 20% chance "invalid"
    If payment is successful, updates order status to "paid".
    """
    logger.info(f"Processing payment: ${amount} via {method} for order {order_id}")
    if order_id not in orders: # Accessing global orders is still fine for tool as it's external data
        logger.error(f"Order {order_id} not found during payment")
        return "order_not_found"

    order_data = orders[order_id] # Accessing global orders
    total_due = order_data["total_cost"]

    if method.lower() == "cash":
        if amount < total_due:
            logger.warning(f"Insufficient cash payment: ${amount} < ${total_due}")
            return "insufficient_funds"
        # Payment success; calculate change
        change = round(amount - total_due, 2)
        # order_data["status"] = "paid" # No longer directly updating global orders
        logger.info(f"Cash payment successful for order {order_id}. Change: ${change}")
        return f"cash_ok with change {change}"

    elif method.lower() == "card":
        # 80% success
        if random.random() < 0.8:
            # order_data["status"] = "paid" # No longer directly updating global orders
            logger.info(f"Card payment successful for order {order_id}")
            return "valid"
        else:
            logger.warning(f"Card payment failed for order {order_id}")
            return "invalid"

    else:
        logger.error(f"Unknown payment method: {method}")
        return "unknown_method"


@tool
def create_order(order_items: OrderItem) -> str:
    """
    Creates a new order with status 'pending', given a list of order items in JSON format,
    for example:
    [
        {"name": "Bruschetta", "quantity": 2},
        {"name": "Lobster Tail", "quantity": 1}
    ]
    Returns the newly created order ID.
    """
    try:
        # Validate order items using Pydantic
        validated_items = [OrderItem(**item) for item in order_items]

        global orders
        logger.info(f"Creating new order with items: {validated_items}")
        order_id = get_new_order_id()

        orders[order_id] = {
            "items": [item.model_dump() for item in validated_items],
            "status": "pending",
            "total_cost": 0.0,
        }
        logger.info(f"Order {order_id} created successfully")
        return f"New order {order_id} created with status 'pending'."

    except ValidationError as e:
        logger.error(f"Order validation error: {str(e)}")
        return f"Error creating order: {str(e)}"


@tool
def process_order(order_id: int) -> str:
    """
    Processes an existing order:
    1. Checks each item's department stock via check_and_update_stock.
    2. If insufficient stock for any item, that item remains pending,
       and the overall order might be partially_fulfilled.
    3. If all items fulfilled, order is marked 'fulfilled'.
    """
    global orders # Still using global orders for simplicity
    logger.info(f"Processing order {order_id}")
    if order_id not in orders: # Accessing global orders
        logger.error(f"Order {order_id} not found")
        return "order_not_found"

    order_data = orders[order_id] # Accessing global orders
    logger.info(f"Order {order_id} status: {order_data['status']}")
    if order_data["status"] not in ["pending", "partially_fulfilled"]:
        logger.warning(f"Cannot process order {order_id} in state: {order_data['status']}")
        return f"cannot_process_order_in_state_{order_data['status']}"

    all_fulfilled = True
    for item in order_data["items"]:
        if item["status"] == "pending":
            # call tool check_and_update_stock item_name, quantity
            logger.info(f"Checking stock for {item['name']} x{item['quantity']} in process_order")
            result = check_and_update_stock(item["name"], item["quantity"])
            logger.info(f"Stock check result for {item['name']}: {result} in process_order")
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
        order_data["status"] = "fulfilled" # Updating global orders
        logger.info(f"Order {order_id} fully fulfilled")
        return f"Order {order_id} is fully fulfilled."
    else:
        # If at least one item remains "pending", the order is partially fulfilled
        order_data["status"] = "partially_fulfilled" # Updating global orders
        logger.warning(f"Order {order_id} partially fulfilled")
        return f"Order {order_id} is partially fulfilled. Some items are not available or still pending."


# -------------------------- LLM and Workflow Initialization -------------------------- #

# Initialize the AzureChatOpenAI LLM
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

AZURE_OPENAI_4OMINI= os.getenv("OPENAI_MODEL_4OMINI")

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
deepseek_llm=ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_ENDPOINT,
    model=DEEPSEEK_MODEL,
    temperature=0.5,
    max_tokens=4096
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_FLASH20= os.getenv("GEMINI_MODEL_FLASH20")
gemini_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_FLASH20, api_key=GEMINI_API_KEY, temperature=0.5, max_tokens=4096,timeout=60,transport="rest")

# Add our expanded tools
tools = [
    get_drinks_menu,
    get_food_menu,
    create_order,
    process_order,
    cashier_calculate_total,
    check_payment,
    get_restaurant_info,
    get_menu_item_price
]

tool_node = ToolNode(tools)
model_with_tools = gemini_llm.bind_tools(tools)

#custom tools condition, can be modified and used in the workflow with workflow.add_conditional_edges if needed
def should_continue(state: RestaurantOrderState):
    messages = state['messages']
    last_message = messages[-1]
    logger.info(f"Last tool message: {last_message}")
    if last_message.tool_calls:
        logger.info("Last message is a tool call")
        return "tools"
    logger.info("Last message is not a tool call")
    return END

def call_model_with_tools(state: RestaurantOrderState):
    messages = state['messages']
    try:
        logger.info(f"LLM input: {messages}")
        response = model_with_tools.invoke(messages)
        logger.info(f"LLM response: {response}")

        return {"messages":[response],
                "order_id": state['order_id'],
                "order_items": state['order_items'],
                "order_status": state['order_status'],
                "total_cost": state['total_cost'],
                "payment_status": state['payment_status'],
                "conversation_rounds": state["conversation_rounds"], 
                "summary": state["summary"]}
    except Exception as e:
        logger.error(f"LLM invoke error: {str(e)}", exc_info=True)
        error_message = AIMessage(content="I apologize, but I'm having trouble processing your request. Could you please try again?")
        return {"messages":[error_message],
                "order_id": state['order_id'],
                "order_items": state['order_items'],
                "order_status": state['order_status'],
                "total_cost": state['total_cost'],
                "payment_status": state['payment_status'],
                "conversation_rounds": state["conversation_rounds"], 
                "summary": state["summary"]}

def route_after_agent(state: RestaurantOrderState) -> str:
    """Decide whether to summarize, use tools, or end conversation"""
    messages = state['messages']
    last_message = messages[-1]

    # If last message was a tool call, continue to tools
    if last_message.tool_calls:
        return "tools"

    # If last message was user input and we need to summarize
    if should_summarize(state):
        return "summarizer"

    # Otherwise end the conversation
    return END


# Build the state graph
workflow = StateGraph(RestaurantOrderState) # Use custom state schema
memory=MemorySaver()
workflow.add_node("agent", call_model_with_tools)
workflow.add_node("tools", tool_node)
workflow.add_node("summarizer", conversation_summarizer)

#the add conditional edges also has problem
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    route_after_agent
)

workflow.add_edge("tools", "agent")
workflow.add_edge("summarizer", "agent")
workflow.add_edge("agent", END)

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Initial system message

def create_initial_state() -> RestaurantOrderState:
    """Create and validate initial state"""

    global system_message

    initial_state = {
        "messages": [system_message]+[HumanMessage(content=" ")],
        "order_id": 0,
        "order_items": [],
        "order_status": "pending",
        "total_cost": 0.0,
        "payment_status": "pending",
        "conversation_rounds": 0,
        "summary": "No summary yet"
    }
    logger.info(f"Initial State Messages: {initial_state['messages']}") # Log initial messages

    return RestaurantOrderState.validate_state(initial_state)


if __name__ == "__main__":
    logger.info("Starting restaurant waiter service with custom state")
    print("Welcome to the restaurant! Type 'q' to quit the conversation.")

    try:
        # Initialize state with validation
        initial_state = create_initial_state()
        result=app.invoke(initial_state, config=config)
        ai_messages = [msg for msg in result['messages'] if isinstance(msg, AIMessage)]
        if ai_messages:
            print("\nWaiter:", ai_messages[-1].content)
            logger.info(f"AI response: {ai_messages[-1].content}")
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == 'q':
                logger.info("User ended conversation")
                print("Thank you for visiting! Goodbye!")
                break

            logger.info(f"User input: {user_input}")
            user_message = HumanMessage(content=user_input, name="user")

            try:
                # Update state with validation
                current_state = app.get_state(config)
                updated_state_dict = {
                    "messages": list(current_state[0]['messages']) + [user_message],
                    "order_id": current_state[0]['order_id'],
                    "order_items": current_state[0]['order_items'],
                    "order_status": current_state[0]['order_status'],
                    "total_cost": current_state[0]['total_cost'],
                    "payment_status": current_state[0]['payment_status'],
                    "conversation_rounds": current_state[0]['conversation_rounds']+1,
                    "summary": current_state[0]['summary']
                }

                validated_state = RestaurantOrderState.validate_state(updated_state_dict)
                result = app.invoke(validated_state, config=config)

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