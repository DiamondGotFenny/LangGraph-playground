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
import re
from pathlib import Path

# Optional .env loading for local development convenience.
# If python-dotenv isn't installed or .env isn't present, we fall back to OS env vars.
# (This is runtime behavior; as a coding agent, I won't read/modify your .env file.)
try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore

    _ = load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Any, List, Dict, Optional, Literal, TypedDict, cast
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    trim_messages,
)
from langchain_core.messages.modifier import RemoveMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START, add_messages
from logger_config import setup_logger
from langgraph.checkpoint.memory import MemorySaver


def ensure_log_file(log_file_path: str) -> str:
    """Ensure log file exists, create if it doesn't, and return the path."""
    log_path = Path(log_file_path)
    try:
        # Create parent directories if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        log_path.touch(exist_ok=True)
        return str(log_path)
    except Exception:
        # Best-effort fallback to current directory without printing to console.
        return "waiter_react_agent.log"  # Fallback to current directory

# Setup logger with guaranteed log file
log_file_path = ensure_log_file("waiter_react_agent.log")
logger = setup_logger(log_file_path, log_to_console=False)


def _render_ai_content(content) -> str:
    """Render AIMessage.content into a human-friendly string (Gemini can return rich parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


REDACT_LOG_KEYS = {
    "__gemini_function_call_thought_signatures__",
    "id",
    "tool_call_id",
    "run_id",
    "parent_run_id",
    "trace_id",
    "span_id",
    "request_id",
    "signature",
}
REDACT_LOG_SUFFIXES = ("_id",)
ALLOW_ID_KEYS = {"order_id"}


def _should_redact_log_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    if key in REDACT_LOG_KEYS:
        return True
    if key in ALLOW_ID_KEYS:
        return False
    return any(key.endswith(suffix) for suffix in REDACT_LOG_SUFFIXES)


def _message_to_dict(msg: Any) -> Any:
    if isinstance(msg, (dict, list)):
        return msg
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump()
        except Exception:
            pass
    if hasattr(msg, "dict"):
        try:
            return msg.dict()
        except Exception:
            pass
    return str(msg)


def _redact_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[Any, Any] = {}
        for key, item in value.items():
            if _should_redact_log_key(key):
                continue
            redacted[key] = _redact_for_log(item)
        if "extras" in redacted and isinstance(redacted["extras"], dict) and not redacted["extras"]:
            redacted.pop("extras", None)
        return redacted
    if isinstance(value, list):
        return [_redact_for_log(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _serialize_for_log(value: Any) -> Any:
    if isinstance(value, list):
        converted = [_message_to_dict(item) for item in value]
    else:
        converted = _message_to_dict(value)
    return _redact_for_log(converted)


def _redact_text(text: str) -> str:
    patterns = (
        (r"\b__gemini_function_call_thought_signatures__\b\s*:\s*\{[^}]*\}", "__gemini_function_call_thought_signatures__: <redacted>"),
        (r"\bid\s+['\"][^'\"]+['\"]", "id '<redacted>'"),
        (r"\btool_call_id\s+['\"][^'\"]+['\"]", "tool_call_id '<redacted>'"),
        (r"\brun_id\s+['\"][^'\"]+['\"]", "run_id '<redacted>'"),
        (r"\bparent_run_id\s+['\"][^'\"]+['\"]", "parent_run_id '<redacted>'"),
        (r"\bsignature\s+['\"][^'\"]+['\"]", "signature '<redacted>'"),
    )
    redacted = text
    for pattern, repl in patterns:
        redacted = re.sub(pattern, repl, redacted)
    return redacted


def _format_messages_for_log(messages: Any) -> str:
    sanitized = _serialize_for_log(messages)
    items = sanitized if isinstance(sanitized, list) else [sanitized]
    lines: List[str] = []
    for idx, msg in enumerate(items, 1):
        if not isinstance(msg, dict):
            lines.append(f"[{idx}] message")
            lines.append(str(msg))
            lines.append("")
            continue
        msg_type = msg.get("type", "message")
        name = msg.get("name")
        header = f"[{idx}] {msg_type}"
        if name:
            header += f" ({name})"
        lines.append(header)

        content = msg.get("content")
        if isinstance(content, list):
            content_text = _render_ai_content(content)
        else:
            content_text = content
        if content_text:
            lines.append(str(content_text))

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict):
                    call_name = call.get("name")
                    args = call.get("args")
                    lines.append(f"tool_call: {call_name} args={args}")
                else:
                    lines.append(f"tool_call: {call}")

        additional_kwargs = msg.get("additional_kwargs")
        if isinstance(additional_kwargs, dict):
            func_call = additional_kwargs.get("function_call")
            if func_call:
                lines.append(f"function_call: {func_call}")

        lines.append("")
    return "\n".join(lines).rstrip()


def _normalize_order_items(order_items: List["OrderItem"]) -> Dict[str, int]:
    """Combine duplicate items and return a name->quantity map."""
    normalized: Dict[str, int] = {}
    for item in order_items:
        normalized[item.name] = normalized.get(item.name, 0) + item.quantity
    return normalized


def _get_active_order_id() -> Optional[int]:
    if ACTIVE_ORDER_ID is None:
        return None
    if ACTIVE_ORDER_ID not in orders:
        return None
    if orders[ACTIVE_ORDER_ID].get("status") == "paid":
        return None
    return ACTIVE_ORDER_ID


def _set_active_order_id(order_id: Optional[int]) -> None:
    global ACTIVE_ORDER_ID
    ACTIVE_ORDER_ID = order_id


def _resolve_order_id(order_id: Optional[int]) -> Optional[int]:
    active_id = _get_active_order_id()
    if active_id is not None and (order_id is None or order_id != active_id):
        return active_id
    if order_id in orders:
        return order_id
    return active_id


def _restock_item(name: str, quantity: int, department: str) -> None:
    if not department or quantity <= 0:
        return
    stock = DEPARTMENT_STOCKS.get(department)
    if stock is None:
        return
    if name not in stock:
        return
    stock[name] += quantity
    logger.info(f"Restocked {name} x{quantity} to {department}")


PRICE_QUESTION_KEYWORDS = ("price", "cost", "how much")


def _extract_menu_item_for_price_question(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    if not any(keyword in lowered for keyword in PRICE_QUESTION_KEYWORDS):
        return None
    matches = [name for name in menu_prices.keys() if name.lower() in lowered]
    if len(matches) == 1:
        return matches[0]
    return None

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
ACTIVE_ORDER_ID: Optional[int] = None
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
        "For each item, provide {\"name\": <str>, \"quantity\": <int>}. For instance: "
        "create_order({\"order_items\": [{\"name\": \"Bruschetta\", \"quantity\": 2}, {\"name\": \"Lobster Tail\", \"quantity\": 1}]})."
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
        "do not return empty in any condition, always return a message to the user."
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

class RestaurantOrderState(TypedDict, total=False):
    """LangGraph state.

    Notes:
    - `messages` uses the `add_messages` reducer, so nodes should return incremental
      updates like `{"messages": [AIMessage(...)]}` or deletions via `RemoveMessage`.
    - We intentionally keep the system prompt *out* of state messages (it is injected
      at model-call time) to keep pruning simple and deterministic.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    summary: str
    conversation_rounds: int


class InputState(TypedDict, total=False):
    messages: List[AnyMessage]


class OutputState(TypedDict, total=False):
    messages: List[AIMessage]


# -------------------------- Tools Section -------------------------- #


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

def _build_summary_prompt(text: str, previous_summary: str) -> List[AnyMessage]:
    summary_instructions = (
        "Summarize the conversation text below.\n"
        "- Focus on concrete details explicitly stated: order items, quantities, preferences, constraints, payment status.\n"
        "- Do NOT invent or infer anything.\n"
        "- Keep it compact and structured.\n"
        "- Return plain text only (no JSON, no list literals).\n"
    )
    if previous_summary:
        summary_instructions += (
            "\nYou also have an existing summary. Update it by incorporating any new information.\n"
            "Existing summary:\n"
            f"{previous_summary}\n"
        )

    return [
        SystemMessage(content=summary_instructions),
        HumanMessage(content=text),
    ]


def _cutoff_index_for_last_n_rounds(messages: List[AnyMessage], *, keep_rounds: int) -> int:
    """Return the index from which we should keep messages to retain last N user/assistant rounds.

    We count HumanMessage/AIMessage pairs as "rounds" (tool messages stay with the kept tail).
    The returned index is a contiguous tail slice start, so tool messages inside those rounds are preserved.
    """
    keep_human_ai = keep_rounds * 2
    seen = 0
    cutoff = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], (HumanMessage, AIMessage)):
            seen += 1
            if seen >= keep_human_ai:
                cutoff = i
                break
    return cutoff


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

    resolved_id = _resolve_order_id(order_id)
    if resolved_id is None or resolved_id not in orders:
        logger.error(f"Order {order_id} not found")
        return 0.0

    order_data = orders[resolved_id]
    subtotal = 0.0
    for item in order_data["items"]:
        name = item["name"]
        qty = item["quantity"]
        price = menu_prices.get(name, 0.0)
        subtotal += price * qty

    tip_amount = subtotal * 0.15
    total = round(subtotal + tip_amount, 2)


    order_data["total_cost"] = total
    order_data["status"] = "billed"
    logger.info(f"Order {resolved_id} total calculated: ${total}")
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
    resolved_id = _resolve_order_id(order_id)
    if resolved_id is None or resolved_id not in orders:
        logger.error(f"Order {order_id} not found during payment")
        return "order_not_found"

    order_data = orders[resolved_id]
    total_due = order_data.get("total_cost", 0.0)
    if total_due <= 0:
        total_due = cashier_calculate_total(resolved_id)

    if method.lower() == "cash":
        if amount < total_due:
            logger.warning(f"Insufficient cash payment: ${amount} < ${total_due}")
            return "insufficient_funds"
        # Payment success; calculate change
        change = round(amount - total_due, 2)
        order_data["status"] = "paid"
        _set_active_order_id(None)
        logger.info(f"Cash payment successful for order {resolved_id}. Change: ${change}")
        return f"cash_ok with change {change}"

    elif method.lower() == "card":
        # 80% success
        if random.random() < 0.8:
            order_data["status"] = "paid"
            _set_active_order_id(None)
            logger.info(f"Card payment successful for order {resolved_id}")
            return "valid"
        else:
            logger.warning(f"Card payment failed for order {resolved_id}")
            return "invalid"

    else:
        logger.error(f"Unknown payment method: {method}")
        return "unknown_method"


class CreateOrderInput(BaseModel):
    order_items: List[OrderItem]


@tool(args_schema=CreateOrderInput)
def create_order(order_items: List[OrderItem]) -> str:
    """
    Creates or updates the active order with status 'pending', given a list of order items in JSON format,
    for example:
    [
        {"name": "Bruschetta", "quantity": 2},
        {"name": "Lobster Tail", "quantity": 1}
    ]
    Returns the newly created or updated order ID.
    """
    try:
        global orders
        incoming_quantities = _normalize_order_items(order_items)
        active_id = _get_active_order_id()
        if active_id is None:
            logger.info(f"Creating new order with items: {order_items}")
            order_id = get_new_order_id()
            _set_active_order_id(order_id)
            orders[order_id] = {
                "items": [
                    {"name": name, "quantity": qty, "department": "", "status": "pending"}
                    for name, qty in incoming_quantities.items()
                ],
                "status": "pending",
                "total_cost": 0.0,
            }
            logger.info(f"Order {order_id} created successfully")
            return f"New order {order_id} created with status 'pending'."

        order_data = orders[active_id]
        existing_items = order_data.get("items", [])
        existing_names = {item["name"] for item in existing_items}
        incoming_names = set(incoming_quantities.keys())
        overlap_ratio = 0.0
        if existing_names:
            overlap_ratio = len(existing_names & incoming_names) / len(existing_names)
        snapshot_mode = overlap_ratio >= 0.6
        logger.info(f"Updating order {active_id} (snapshot={snapshot_mode}) with items: {order_items}")

        if not snapshot_mode:
            new_items = list(existing_items)
            for name, qty in incoming_quantities.items():
                pending_entry = next(
                    (item for item in new_items if item["name"] == name and item["status"] == "pending"),
                    None,
                )
                if pending_entry:
                    pending_entry["quantity"] += qty
                else:
                    new_items.append(
                        {"name": name, "quantity": qty, "department": "", "status": "pending"}
                    )
            order_data["items"] = new_items
        else:
            existing_by_name: Dict[str, List[Dict[str, object]]] = {}
            for item in existing_items:
                existing_by_name.setdefault(item["name"], []).append(item)

            new_items: List[Dict[str, object]] = []
            for name, desired_qty in incoming_quantities.items():
                current_entries = existing_by_name.get(name, [])
                current_total = sum(item["quantity"] for item in current_entries)
                if not current_entries:
                    new_items.append(
                        {"name": name, "quantity": desired_qty, "department": "", "status": "pending"}
                    )
                    continue
                if desired_qty >= current_total:
                    new_items.extend(current_entries)
                    delta = desired_qty - current_total
                    if delta > 0:
                        new_items.append(
                            {"name": name, "quantity": delta, "department": "", "status": "pending"}
                        )
                    continue

                remaining_to_remove = current_total - desired_qty
                adjusted_entries: List[Dict[str, object]] = []
                for entry in sorted(
                    current_entries, key=lambda e: 0 if e["status"] == "pending" else 1
                ):
                    if remaining_to_remove <= 0:
                        adjusted_entries.append(entry)
                        continue
                    entry_qty = entry["quantity"]
                    if entry_qty <= remaining_to_remove:
                        if entry["status"] == "fulfilled":
                            _restock_item(name, entry_qty, cast(str, entry.get("department", "")))
                        remaining_to_remove -= entry_qty
                        continue
                    if entry["status"] == "fulfilled":
                        _restock_item(name, remaining_to_remove, cast(str, entry.get("department", "")))
                    updated_entry = dict(entry)
                    updated_entry["quantity"] = entry_qty - remaining_to_remove
                    remaining_to_remove = 0
                    adjusted_entries.append(updated_entry)
                new_items.extend(adjusted_entries)

            removed_names = existing_names - incoming_names
            for name in removed_names:
                for entry in existing_by_name.get(name, []):
                    if entry["status"] == "fulfilled":
                        _restock_item(name, entry["quantity"], cast(str, entry.get("department", "")))

            order_data["items"] = new_items

        pending_exists = any(item["status"] == "pending" for item in order_data["items"])
        fulfilled_exists = any(item["status"] == "fulfilled" for item in order_data["items"])
        if pending_exists and fulfilled_exists:
            order_data["status"] = "partially_fulfilled"
        elif pending_exists:
            order_data["status"] = "pending"
        elif fulfilled_exists:
            order_data["status"] = "fulfilled"
        else:
            order_data["status"] = "pending"

        return f"Order {active_id} updated with status '{order_data['status']}'."

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
    resolved_id = _resolve_order_id(order_id)
    if resolved_id is None or resolved_id not in orders:
        logger.error(f"Order {order_id} not found")
        return "order_not_found"

    order_data = orders[resolved_id]
    logger.info(f"Order {resolved_id} status: {order_data['status']}")
    if order_data["status"] not in ["pending", "partially_fulfilled"]:
        logger.warning(f"Cannot process order {resolved_id} in state: {order_data['status']}")
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
        logger.info(f"Order {resolved_id} fully fulfilled")
        return f"Order {resolved_id} is fully fulfilled."
    else:
        # If at least one item remains "pending", the order is partially fulfilled
        order_data["status"] = "partially_fulfilled" # Updating global orders
        logger.warning(f"Order {resolved_id} partially fulfilled")
        return f"Order {resolved_id} is partially fulfilled. Some items are not available or still pending."


# -------------------------- LLM and Workflow Initialization -------------------------- #

def _build_llm():
    """Select a single chat model based on env vars.

    Priority: Gemini -> DeepSeek -> Azure OpenAI.

    Notes:
    - We rely on a recent `langchain-google-genai` + `google-genai` SDK that supports
      Gemini 3 thought signatures for tool/function calling. We do NOT implement
      fallbacks to older Gemini models here.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = (
        os.getenv("GEMINI_MODEL_FLASH30")
        or os.getenv("GEMINI_MODEL_FLASH20")
        or os.getenv("GEMINI_MODEL_FLASH15")
    )
    if gemini_api_key and gemini_model:
        logger.info(f"Using Gemini model: {gemini_model}")
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=gemini_api_key,
            temperature=1,
            max_tokens=4096,
            timeout=60,
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_endpoint = os.getenv("DEEPSEEK_ENDPOINT")
    deepseek_model = os.getenv("DEEPSEEK_MODEL")
    if deepseek_api_key and deepseek_endpoint and deepseek_model:
        logger.info(f"Using DeepSeek model: {deepseek_model}")
        return ChatOpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_endpoint,
            model=deepseek_model,
            temperature=0.5,
            max_tokens=4096,
        )

    azure_api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("OPENAI_MODEL_4o")
    azure_api_version = os.getenv("AZURE_API_VERSION")
    if azure_api_key and azure_endpoint and azure_deployment and azure_api_version:
        logger.info(f"Using Azure OpenAI deployment: {azure_deployment}")
        return AzureChatOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            deployment_name=azure_deployment,
            api_version=azure_api_version,
            temperature=0.5,
            max_tokens=4000,
        )

    missing_hint = (
        "No LLM is configured.\n\n"
        "Set ONE of the following env-var sets (PowerShell examples):\n"
        "- Gemini:\n"
        "  $env:GEMINI_API_KEY=\"...\"; $env:GEMINI_MODEL_FLASH30=\"gemini-2.0-flash\"; python waiter_react_agent.py\n"
        "  (or use GEMINI_MODEL_FLASH20 / GEMINI_MODEL_FLASH15)\n"
        "- DeepSeek:\n"
        "  $env:DEEPSEEK_API_KEY=\"...\"; $env:DEEPSEEK_ENDPOINT=\"...\"; $env:DEEPSEEK_MODEL=\"...\"; python waiter_react_agent.py\n\n"
        "- Azure OpenAI:\n"
        "  $env:OPENAI_API_KEY=\"...\"; $env:AZURE_OPENAI_ENDPOINT=\"...\"; $env:OPENAI_MODEL_4o=\"...\"; $env:AZURE_API_VERSION=\"...\"; python waiter_react_agent.py\n"
        "Detected presence (True/False):\n"
        f"- GEMINI_API_KEY: {bool(gemini_api_key)}; GEMINI_MODEL_FLASH30/20/15: {bool(gemini_model)}\n"
        f"- DEEPSEEK_API_KEY: {bool(deepseek_api_key)}; DEEPSEEK_ENDPOINT: {bool(deepseek_endpoint)}; DEEPSEEK_MODEL: {bool(deepseek_model)}\n"
        f"- OPENAI_API_KEY: {bool(azure_api_key)}; AZURE_OPENAI_ENDPOINT: {bool(azure_endpoint)}; OPENAI_MODEL_4o: {bool(azure_deployment)}; AZURE_API_VERSION: {bool(azure_api_version)}\n"
    )
    raise RuntimeError(missing_hint)


# Add our expanded tools
tools = [
    get_drinks_menu,
    get_food_menu,
    create_order,
    process_order,
    cashier_calculate_total,
    check_payment,
    get_restaurant_info,
    get_menu_item_price,
    # conversation_summarizer, # Removed summarizer from tools
]

tool_node = ToolNode(tools)
model_with_tools = None  # initialized at runtime in __main__

#custom tools condition, can be modified and used in the workflow with workflow.add_conditional_edges if needed

def _agent_node(state: RestaurantOrderState) -> Dict:
    messages = cast(List[AnyMessage], state.get("messages", []))
    summary = cast(str, state.get("summary", ""))
    conversation_rounds = int(state.get("conversation_rounds", 0))

    if messages and isinstance(messages[-1], HumanMessage):
        conversation_rounds += 1
        price_item = _extract_menu_item_for_price_question(messages[-1].content)
        if price_item:
            price_text = get_menu_item_price.invoke({"item_name": price_item})
            response_text = f"{price_text} Would you like anything else?"
            return {"messages": [AIMessage(content=response_text)], "conversation_rounds": conversation_rounds}

    # Some providers (notably Gemini) reject requests with no "contents". If we have no
    # user/assistant history yet, don't call the model—return a deterministic greeting.
    if not messages and not summary:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Welcome to Villa Toscana! I'm delighted to have you with us today. "
                        "Would you like to see the food or drinks menu, or can I recommend something to start?"
                    )
                )
            ],
            "conversation_rounds": 0,
        }

    # Keep model input small and valid (tool messages must follow their AI tool call).
    # We only store convo messages in state; system prompt is injected here.
    model_messages: List[AnyMessage] = [system_message]
    if summary:
        model_messages.append(SystemMessage(content=f"Conversation summary so far:\n{summary}"))

    # Also trim *input* to the model (independent of persisted state pruning).
    # Use message-count trimming to avoid accidental token overflow while keeping structure valid.
    trimmed_for_model = trim_messages(
        messages,
        max_tokens=50,  # count messages, not tokens
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )

    # Safety: ensure we never send a system-only request.
    if not trimmed_for_model:
        trimmed_for_model = [HumanMessage(content="Please greet the customer and ask how you can help.")]
    model_messages.extend(trimmed_for_model)

    try:
        formatted_input = _format_messages_for_log(model_messages)
        logger.info(f"LLM input (len={len(model_messages)}):\n{formatted_input}")
        if model_with_tools is None:
            raise RuntimeError("LLM is not initialized. Start the script via __main__.")
        response = model_with_tools.invoke(model_messages)
        formatted_response = _format_messages_for_log(response)
        logger.info(f"LLM response:\n{formatted_response}")
        return {"messages": [response], "conversation_rounds": conversation_rounds}
    except Exception as e:
        logger.error(f"LLM invoke error: {str(e)}", exc_info=True)
        error_message = AIMessage(
            content="I apologize, but I'm having trouble processing your request. Could you please try again?"
        )
        return {"messages": [error_message], "conversation_rounds": conversation_rounds}


def _postprocess_node(state: RestaurantOrderState) -> Dict:
    """Summarize + prune persisted history to only keep the last 5 rounds."""
    messages = cast(List[AnyMessage], state.get("messages", []))
    previous_summary = cast(str, state.get("summary", ""))

    keep_rounds = 5
    cutoff = _cutoff_index_for_last_n_rounds(messages, keep_rounds=keep_rounds)
    if cutoff <= 0:
        return {}

    to_summarize = messages[:cutoff]
    to_keep = messages[cutoff:]

    # Build deletions using RemoveMessage so add_messages reducer removes them by id.
    deletions: List[RemoveMessage] = []
    for m in to_summarize:
        if m.id is not None:
            deletions.append(RemoveMessage(id=m.id))

    # Summarize ONLY the portion we're removing.
    # Exclude raw tool payload verbosity but keep meaning by stringifying the messages.
    text = get_buffer_string(to_summarize, human_prefix="User", ai_prefix="Waiter")
    summary_prompt = _build_summary_prompt(text=text, previous_summary=previous_summary)
    try:
        if model_with_tools is None:
            raise RuntimeError("LLM is not initialized. Start the script via __main__.")
        # Tool-bound chat models still support invoke() for normal messages.
        raw_summary = model_with_tools.invoke(summary_prompt).content
        new_summary = _render_ai_content(raw_summary).strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
        new_summary = previous_summary  # best effort; still prune

    # Reset rounds to the kept window size (or less if conversation is shorter).
    kept_rounds = 0
    for msg in to_keep:
        if isinstance(msg, HumanMessage):
            kept_rounds += 1
    kept_rounds = min(kept_rounds, keep_rounds)

    return {"messages": deletions, "summary": new_summary, "conversation_rounds": kept_rounds}


# Build the state graph
workflow = StateGraph(RestaurantOrderState, input_schema=InputState, output_schema=OutputState)
memory=MemorySaver()
workflow.add_node("agent", _agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("postprocess", _postprocess_node)

workflow.add_edge(START, "agent")

def _route_after_agent(state: RestaurantOrderState) -> str:
    msgs = cast(List[AnyMessage], state.get("messages", []))
    if not msgs:
        return "postprocess"
    last = msgs[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "postprocess"

workflow.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "postprocess": "postprocess"})

workflow.add_edge("tools", "agent")
workflow.add_edge("postprocess", END)

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Initial system message

if __name__ == "__main__":
    logger.info("Starting restaurant waiter service with custom state")
    print("Welcome to the restaurant! Type 'q' to quit the conversation.")

    try:
        # Initialize model at runtime so missing env-vars don't print a full traceback on import.
        llm = _build_llm()
        model_with_tools = llm.bind_tools(tools)

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == 'q':
                logger.info("User ended conversation")
                print("Thank you for visiting! Goodbye!")
                break

            logger.info(f"User input: {user_input}")
            user_message = HumanMessage(content=user_input, name="user")

            # IMPORTANT: only pass the *new* message. `add_messages` + checkpointer will persist history.
            result = app.invoke({"messages": [user_message]}, config=config)
            ai_messages = [msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)]
            if ai_messages:
                rendered = _render_ai_content(ai_messages[-1].content)
                print("\nWaiter:", rendered)
                logger.info(f"AI response: {rendered}")
                logger.info("")
                logger.info("------------------------------------")
                logger.info("")

    except RuntimeError as e:
        # Configuration issues (e.g., missing API keys) should not dump a full traceback to the terminal.
        logger.error(str(e), exc_info=True)
        print(str(e))
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        print("\nWaiter: I apologize, but our system is currently unavailable. Please try again later.")
