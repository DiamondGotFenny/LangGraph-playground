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

from __future__ import annotations

import json
import os
import random
import re
import copy
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
from langchain_core.tools import InjectedToolCallId
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
    trim_messages,
)
from langchain_core.messages.modifier import RemoveMessage
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph import StateGraph, END, START, add_messages
from logger_config import setup_logger
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


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


def _normalize_special_instructions(value: Optional[str]) -> str:
    if value is None:
        return ""
    value = " ".join(str(value).split()).strip()
    return value.lower()


def _order_item_key(name: str, special_instructions: Optional[str]) -> tuple[str, str]:
    return (name, _normalize_special_instructions(special_instructions))


def _normalize_order_items(order_items: List["OrderItem"]) -> Dict[tuple[str, str], int]:
    """Combine duplicate items by (name, special_instructions) and return a key->quantity map."""
    normalized: Dict[tuple[str, str], int] = {}
    for item in order_items:
        key = _order_item_key(item.name, item.special_instructions)
        normalized[key] = normalized.get(key, 0) + item.quantity
    return normalized


def _get_active_order_id_from_state(state: RestaurantOrderState) -> Optional[int]:
    orders_state = cast(Dict[int, Dict[str, Any]], state.get("orders", {}))
    active_id = cast(Optional[int], state.get("active_order_id"))
    if active_id is None:
        return None
    if active_id not in orders_state:
        return None
    if orders_state[active_id].get("status") == "paid":
        return None
    return active_id


def _resolve_order_id_from_state(state: RestaurantOrderState, order_id: Optional[int]) -> Optional[int]:
    orders_state = cast(Dict[int, Dict[str, Any]], state.get("orders", {}))
    active_id = _get_active_order_id_from_state(state)
    if active_id is not None and (order_id is None or order_id != active_id):
        return active_id
    if order_id is not None and order_id in orders_state:
        return order_id
    return active_id


def _restock_item_in_stocks(stocks: Dict[str, Dict[str, int]], name: str, quantity: int, department: str) -> None:
    if not department or quantity <= 0:
        return
    dept_stock = stocks.get(department)
    if dept_stock is None:
        return
    if name not in dept_stock:
        return
    dept_stock[name] = int(dept_stock[name]) + int(quantity)
    logger.info(f"Restocked {name} x{quantity} to {department}")

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

# NOTE: We do NOT keep mutable global conversation state.
# Orders, current active order id, counters, and per-department stock are persisted in LangGraph state
# (per `thread_id`) via InjectedState + Command(update=...).
# Structured System Prompt (v2.0 - Fixed based on Golden Dataset test results)
system_message = SystemMessage(
    content="""You are a waiter at Villa Toscana, an upscale Italian restaurant. Provide excellent service by managing orders, answering questions, and coordinating with kitchen departments.

═══════════════════════════════════════════════════════════════
CRITICAL: ANTI-HALLUCINATION PROTOCOL
═══════════════════════════════════════════════════════════════

NEVER invent, fabricate, or assume information. ALWAYS verify through tools:

1. MENU ITEMS:
   - NEVER guess or invent menu items from memory
   - ALWAYS call get_food_menu or get_drinks_menu to verify items
   - When recommending alternatives, ONLY suggest items from tool results
   
2. PRICES:
   - NEVER quote prices without tool confirmation
   - ALWAYS call get_menu_item_price when discussing specific prices
   
3. STOCK AVAILABILITY:
   - When process_order returns "insufficient_stock":
     * Tell customer how many are available
     * Call get_food_menu or get_drinks_menu to find alternatives from the same category
     * Recommend items that are similar to the unavailable item
   - NEVER recommend items not confirmed by tools

4. ORDER STATUS:
   - Trust tool results (process_order, cashier_calculate_total, check_payment)
   - NEVER assume order status; always reference the latest tool response

5. RESTAURANT INFO:
   - For history, hours, awards → call get_restaurant_info
   - NEVER fabricate restaurant details

Golden Rule: When in doubt, call the appropriate tool. Accuracy > Speed.

═══════════════════════════════════════════════════════════════
CRITICAL: TOOL RESULT PROCESSING PROTOCOL (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════

After EVERY tool call, you MUST follow this exact sequence:

1. READ the complete tool response JSON
2. CHECK for error indicators:
   - "ok": false
   - "status": "insufficient_stock" | "partially_fulfilled" | "item_not_found"
   - "error": any error message
   
3. IF error detected → STOP normal flow and handle error:
   
   a) insufficient_stock:
      - Extract: available quantity vs requested quantity
      - Tell customer: "I apologize, but we only have X [item] available, not the Y you requested."
      - If stock is zero or customer wants alternatives:
        * Call get_food_menu or get_drinks_menu to see what's available
        * Recommend similar items from the same category (e.g., desserts for dessert, pasta for pasta)
      - Example: "Would you like all X we have, or would you prefer to try [similar verified item] instead?"
      - NEVER invent or guess alternatives; always verify through tools first
      - NEVER say "added" or "ordered" or "will bring" if stock insufficient
   
   b) partially_fulfilled:
      - Identify which items succeeded and which failed
      - Explain specifically: "I was able to add [successful items], but [failed items] are not available."
      - Ask: "Would you like to modify or choose alternatives?"
   
   c) cannot_process_order_in_state_X:
      - Explain: "Your order is already in [state], so I cannot process it again."
      - Offer correct action: "Let me [update/finalize/etc.] instead."
   
   d) ANY "ok": false:
      - NEVER claim success ("done", "processed", "added") when tool returned error
      - Explain the specific problem
      - Offer clear next steps

4. VERIFICATION before responding:
   - Ask yourself: "Did the tool report success or failure?"
   - If uncertain, re-read the tool response
   - Tool truth ALWAYS overrides your assumptions

Example - WRONG ❌:
  Tool: {"status": "insufficient_stock", "available": 5, "requested": 7}
  You: "Great! I've added 7 Chocolate Cakes!"

Example - CORRECT ✅:
  Step 1 - Tool: {"status": "insufficient_stock", "available": 5, "requested": 7}
  Step 2 - [Call get_food_menu to check dessert alternatives]
  Step 3 - You: "I apologize, but we only have 5 Chocolate Cakes available today, not the 7 you requested. Would you like all 5, or would you prefer our Cheesecake or Tiramisu instead?"

═══════════════════════════════════════════════════════════════
CRITICAL: PRICE INFORMATION PROTOCOL
═══════════════════════════════════════════════════════════════

Goal: Provide accurate prices while minimizing redundant tool calls.

When user asks about a specific item's price, follow this decision tree:

STEP 1: Check your sources in order of reliability

  Priority 1 - Recent explicit context (HIGHEST trust):
  ✓ Full menu with prices shown in last 5 messages
  ✓ Price tool result in last 3 messages
  → Action: Reference directly with acknowledgment
  
  Priority 2 - Memory/Summary (MEDIUM trust):
  ✓ Your conversation summary contains: "told customer [item] costs $X"
  ✓ Earlier in conversation (beyond 5 messages) you stated a price
  → Action: Reference with confidence
  → Optional: Offer verification if user seems uncertain
  
  Priority 3 - No reliable source (MUST call tool):
  ✗ No price info in recent context, memory, or summary
  → Action: Call get_menu_item_price(item_name) immediately
  → Then quote the result

STEP 2: Apply the Golden Rule

  ✓ Use ANY verifiable source (recent context, memory/summary, tool)
  ✗ NEVER guess from training data or "typical restaurant prices"
  ✗ NEVER estimate or say "around $X" or "probably $X"
  ✗ If uncertain about source reliability → call tool (safe choice)

STEP 3: Trust the safety net

  - Final billing (cashier_calculate_total) always uses correct prices
  - Your job: be as accurate as possible in conversation phase
  - Small discrepancies will be corrected at checkout (but avoid them)

Examples:

Example 1 - Menu in recent context ✅:
  [2 messages ago: Menu showed "Filet Mignon $35.00"]
  User: "How much is the Filet Mignon?"
  You: "As shown in our menu, the Filet Mignon is $35.00"
  (No tool call needed - efficient!)

Example 2 - Price in memory/summary ✅:
  [Summary: "told customer Filet Mignon is $35.00"]
  User: "Remind me, how much was the Filet Mignon?"
  You: "As I mentioned earlier, the Filet Mignon is $35.00"
  (No tool call needed - use memory!)

Example 3 - No reliable source ✅:
  [No menu shown, no memory of this item]
  User: "How much is the Filet Mignon?"
  [Call: get_menu_item_price("Filet Mignon")]
  Tool: "Filet Mignon costs $35.00"
  You: "The Filet Mignon is $35.00"
  (Tool call required - ensure accuracy!)

Example 4 - WRONG ❌:
  [No verifiable source]
  User: "How much is the Filet Mignon?"
  You: "It's $36.00" (guessed from training data)
  (Violation! Must call tool when no source!)

═══════════════════════════════════════════════════════════════
TOOL CALLING DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════

Before calling ANY tool, check context first.

1. MENU TOOLS (get_food_menu, get_drinks_menu):

   *** CONTEXT-FIRST RULE (CHECK BEFORE CALLING) ***:
   Before calling menu tools, ask yourself:
   - "Is the full menu already visible in the last 5 messages?"
   - "Did user explicitly request to see the menu?"
   If menu already in context → DO NOT CALL, reference it directly.

   CALL ONLY IF ALL are true:
   ✓ User explicitly asks: "show menu", "can I see menu", "what's on the menu"
   ✓ Full menu NOT in recent conversation (last 5 messages)
   ✓ You need to verify item names for recommendations or alternatives

   DO NOT CALL IF:
   ✗ Menu already shown in recent context → Say "Looking at our menu..." and reference it
   ✗ User asks single item price only → Use get_menu_item_price instead
   ✗ After calling get_restaurant_info → User asked about restaurant, not menu
   ✗ Verifying well-known items (e.g., "Cappuccino", "Steak") → Call tool only when needed for alternatives

   COORDINATION with Price Protocol:
   - If menu in context contains prices → Reference those prices per Price Protocol above
   - Single price query without menu → Use get_menu_item_price (not full menu)
   - User wants "all prices" or "complete menu" → Use get_food_menu/get_drinks_menu

   TIP: Menu tools are primarily for displaying to customer or finding alternatives, not for internal verification.

2. PRICE TOOL (get_menu_item_price):
   
   CALL IF (MANDATORY):
   ✓ User asks "how much is X"
   ✓ EVERY SINGLE TIME (never skip, even if you just checked)
   ✓ Before writing ANY numeric price in your response
   
   NEVER: Quote prices without calling tool (see Price Protocol above)

3. ORDER TOOLS (create_order, process_order):
   
   create_order:
   - CALL: When user confirms items to order
   - VERIFY: Convert colloquial names to official menu names first
     (e.g., "steak" → ask which: Filet Mignon or Ribeye Steak)
     (e.g., "iced tea" → "Black Tea")
   - FORMAT: [{"name": "Item Name", "quantity": 2, "special_instructions": "medium rare"}]
   
   process_order:
   - CHECK order status from previous tool result FIRST
   - CALL ONLY IF: Status is 'pending' OR 'partially_fulfilled'
   - DO NOT CALL IF: Status is 'fulfilled', 'billed', or 'paid'
   - IF tool returns error → Follow CRITICAL protocol above (don't claim success)

4. PAYMENT TOOLS (cashier_calculate_total, check_payment):
   
   CALL ONLY IF user explicitly says:
   ✓ "check please" / "bill please"
   ✓ "I'll pay now" / "can I pay"
   ✓ "how much do I owe"
   ✓ "I'm done" (clearly finished ordering)
   
   DO NOT:
   ✗ Suggest payment before user asks
   ✗ Call if order not yet fulfilled
   
   IF payment fails:
   → Ask politely for alternative payment method
   → If fails again, offer to get manager assistance

5. RESTAURANT INFO TOOL (get_restaurant_info):
   CALL IF: User asks about history, awards, hours, team, accolades
   DO NOT CALL: For menu or food recommendations

6. CUSTOMER PROFILE TOOL (update_customer_profile):
   CALL IF: User shares name, allergies, dietary restrictions, preferences, or important notes.
   - Examples: "I'm allergic to nuts", "no dairy", "I like my steak medium-rare", "my name is Alex"
   DO NOT CALL: For ordering items (use create_order/process_order), or for prices/menus.

═══════════════════════════════════════════════════════════════
SERVICE STANDARDS
═══════════════════════════════════════════════════════════════

✓ Always be polite, professional, and attentive
✓ Confirm orders by repeating items back to customer
✓ Handle small talk briefly, then guide back to service
✓ After completing requests, ask "Is there anything else I can help you with?"
✓ ALWAYS respond with a message (never empty response)

═══════════════════════════════════════════════════════════════
CONVERSATION FLOW
═══════════════════════════════════════════════════════════════

1. Greeting → Offer menu or recommendations
2. Take order → Verify names, call create_order then process_order
3. Handle modifications → Update order via create_order
4. When customer signals done → Wait for payment request
5. Process payment → Thank customer and bid farewell

Remember: Tool results are FACTS. Never contradict them. When tools report problems, help customer solve them - don't pretend they don't exist.
"""
)

def get_new_order_id() -> int:
    """Deprecated: order ids are tracked in LangGraph state now.

    Kept only to avoid breaking external imports; do not use inside this module.
    """
    raise RuntimeError("get_new_order_id is deprecated; use state['order_counter'] inside tools instead.")


# -------------------------- LangGraph State Definition -------------------------- #
class OrderItem(BaseModel):
    """Represents a single item in an order"""
    name: str
    quantity: int = Field(ge=0)  # Allow 0 to support removals in order updates
    special_instructions: Optional[str] = Field(
        default=None,
        description="Optional customer notes (e.g., steak doneness, no ice, allergies).",
    )
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
        if not isinstance(value, int) or value < 0:
            raise ValueError("Quantity must be a non-negative integer")
        return value

    @field_validator("special_instructions")
    @classmethod
    def normalize_special_instructions(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = " ".join(str(value).split()).strip()
        return value or None

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

    # Persisted structured facts (per thread_id via checkpointer)
    customer: Dict[str, Any]
    orders: Dict[int, Dict[str, Any]]
    active_order_id: Optional[int]
    order_counter: int
    department_stocks: Dict[str, Dict[str, int]]


class InputState(TypedDict, total=False):
    messages: List[AnyMessage]


class OutputState(TypedDict, total=False):
    messages: List[AnyMessage]


def _init_state_node(state: RestaurantOrderState) -> Dict[str, Any]:
    """Initialize structured state fields once per thread."""
    updates: Dict[str, Any] = {}

    if "orders" not in state:
        updates["orders"] = {}
    if "active_order_id" not in state:
        updates["active_order_id"] = None
    if "order_counter" not in state:
        updates["order_counter"] = 0
    if "department_stocks" not in state:
        updates["department_stocks"] = {dept: dict(items) for dept, items in initial_stocks.items()}
    if "customer" not in state:
        updates["customer"] = {
            "name": None,
            "allergies": [],
            "dietary_restrictions": [],
            "preferences": {},
            "notes": "",
        }

    return updates


# -------------------------- Tools Section -------------------------- #


@tool
def get_drinks_menu() -> str:
    """Get the complete drinks menu with all available beverages.
    
    WHEN TO CALL:
    - User explicitly requests drinks menu ("show me drinks", "what drinks do you have")
    - You need to find alternatives when an item is unavailable
    
    DO NOT CALL:
    - If drinks menu already shown in recent conversation (check context first)
    - For restaurant information (use get_restaurant_info instead)
    """
    logger.info("Tool get_drinks_menu called")
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
    """Get the complete food menu with all available dishes.
    
    WHEN TO CALL:
    - User explicitly requests food menu ("show me the menu", "what food do you have")
    - You need to find alternatives when an item is unavailable
    - User asks about options for ambiguous terms (e.g., "steak" → show available steaks)
    
    DO NOT CALL:
    - If food menu already shown in recent conversation (check context first)
    - For restaurant information (use get_restaurant_info instead)
    
    TIP: Always check conversation history before calling. If menu visible, reference it directly.
    """
    logger.info("Tool get_food_menu called")
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
    """Get the current price of a specific menu item.
    
    WHEN TO CALL:
    - User asks "how much is [item]?" AND you have NO reliable source for the price
    - No recent menu display (last 5 messages) containing this price
    - No memory/summary mentioning this specific price
    - When you need to verify/confirm a price you're uncertain about
    
    DO NOT CALL (efficiency optimization):
    - If full menu with prices just shown in recent conversation (last 5 messages)
      → Reference that menu directly: "As shown in our menu, [item] is $X"
    - If your memory/summary contains this price from earlier conversation
      → Reference memory: "As I mentioned earlier, [item] is $X"
    - If user asks for complete price list (use get_food_menu/get_drinks_menu instead)
    
    CRITICAL RULE:
    - If NO verifiable source (context or memory) → MUST call this tool
    - Never guess prices from training data or "typical restaurant prices"
    
    NOTE: Final billing (cashier_calculate_total) always uses actual menu prices,
    so small conversation inaccuracies will be corrected at checkout. But still
    strive for accuracy to maintain customer trust.
    
    Args:
        item_name: The exact menu item name (e.g., "Filet Mignon", "Black Tea")
    
    Returns:
        Price string if found, or error message if item not in menu
    """
    logger.info(f"Tool get_menu_item_price called: item_name={item_name}")
    if item_name in menu_prices:
        return f"{item_name} costs ${menu_prices[item_name]:.2f}"
    return f"Item '{item_name}' not found in menu"

@tool
def get_restaurant_info() -> str:
    """Get detailed information about Villa Toscana restaurant.
    
    WHEN TO CALL:
    - User asks about restaurant history, background, or "tell me about this restaurant"
    - User asks about hours of operation or when restaurant is open
    - User asks about awards, accolades, or recognition
    - User asks about the chef, owner, or team
    
    DO NOT CALL:
    - For menu or food/drink information (use menu tools)
    - For food recommendations (answer from knowledge)

    NOTE:
    - After calling this tool, do NOT automatically call get_food_menu/get_drinks_menu unless the user explicitly asks.
    """
    logger.info("Tool get_restaurant_info called")
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

    ---
    [AGENT NOTE: User asked about restaurant information, NOT the menu. 
     Do NOT automatically call get_food_menu or get_drinks_menu unless user explicitly requests them.]
    """


@tool
def update_customer_profile(
    name: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    dietary_restrictions: Optional[List[str]] = None,
    preferences: Optional[Dict[str, str]] = None,
    notes: Optional[str] = None,
    *,
    state: Annotated[RestaurantOrderState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Persist structured customer facts to graph state (per thread).

    Use this when the user states personal preferences or constraints so the agent can
    reliably remember them without re-parsing free text.
    """
    if not isinstance(state, dict):
        return Command(
            update={
                "messages": [
                    ToolMessage(content="customer_profile_update_failed: missing_state", tool_call_id=tool_call_id)
                ]
            }
        )

    current = state.get("customer")
    profile: Dict[str, Any] = copy.deepcopy(current) if isinstance(current, dict) else {}

    if isinstance(name, str) and name.strip():
        profile["name"] = name.strip()

    def _merge_list(key: str, values: Optional[List[str]]) -> None:
        if not values:
            return
        existing = profile.get(key)
        merged: List[str] = []
        if isinstance(existing, list):
            merged.extend([str(v).strip() for v in existing if str(v).strip()])
        merged.extend([str(v).strip() for v in values if str(v).strip()])
        # de-dupe while preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for v in merged:
            low = v.lower()
            if low in seen:
                continue
            seen.add(low)
            deduped.append(v)
        profile[key] = deduped

    _merge_list("allergies", allergies)
    _merge_list("dietary_restrictions", dietary_restrictions)

    if isinstance(preferences, dict) and preferences:
        existing_prefs = profile.get("preferences")
        prefs_out: Dict[str, str] = {}
        if isinstance(existing_prefs, dict):
            for k, v in existing_prefs.items():
                if isinstance(k, str) and isinstance(v, str):
                    prefs_out[k] = v
        for k, v in preferences.items():
            if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
                prefs_out[k.strip()] = v.strip()
        profile["preferences"] = prefs_out

    if isinstance(notes, str) and notes.strip():
        existing_notes = profile.get("notes")
        if isinstance(existing_notes, str) and existing_notes.strip():
            profile["notes"] = (existing_notes.strip() + " " + notes.strip()).strip()
        else:
            profile["notes"] = notes.strip()

    return Command(
        update={
            "customer": profile,
            "messages": [ToolMessage(content="customer_profile_updated", tool_call_id=tool_call_id)],
        }
    )


def check_and_update_stock(
    department_stocks: Dict[str, Dict[str, int]], item_name: str, quantity: int
) -> Dict[str, Any]:
    """
    Check if the specified item is in stock in the relevant department.
    If in stock, decrement the inventory and return 'fulfilled'.
    If not enough stock, return 'insufficient_stock'.
    If item not found, return 'item_not_found'.
    """
    logger.info(f"Checking stock for item: {item_name}, quantity: {quantity}")
    for department, items in department_stocks.items():
        if item_name in items:
            available_before = int(items[item_name])
            if available_before >= quantity:
                items[item_name] = available_before - quantity
                logger.info(f"Stock fulfilled: {item_name} - {quantity} from {department}")
                return {
                    "status": "fulfilled",
                    "department": department,
                    "requested": quantity,
                    "available_before": available_before,
                    "available_after": int(items[item_name]),
                }
            logger.warning(
                f"Insufficient stock for {item_name} in {department}: requested={quantity} available={available_before}"
            )
            return {
                "status": "insufficient_stock",
                "department": department,
                "requested": quantity,
                "available": available_before,
                "available_before": available_before,
                "available_after": available_before,
            }
    logger.error(f"Item not found in any department: {item_name}")
    return {"status": "item_not_found", "requested": quantity}

def _build_summary_prompt(text: str, previous_summary: str) -> List[AnyMessage]:
    summary_instructions = (
        "Summarize the conversation text below.\n"
        "- Focus on concrete details explicitly stated: order items, quantities, preferences, constraints, payment status.\n"
        "- Do NOT invent or infer anything.\n"
        "\n"
        "*** ORDER STATUS AUTHORITY RULES (CRITICAL) ***\n"
        "Order status MUST be determined by the LATEST authoritative tool result in this priority:\n"
        "\n"
        "  1. If ORDER_SNAPSHOT is provided below, use its status as the PRIMARY source.\n"
        "\n"
        "  2. Tool authority hierarchy (higher overrides lower):\n"
        "     a) check_payment success (cash_ok/valid) → status = 'paid'\n"
        "     b) cashier_calculate_total success → status = 'billed'\n"
        "     c) process_order result → use its 'order_status' field (fulfilled/partially_fulfilled)\n"
        "     d) create_order result → ONLY use if no process_order ran after it\n"
        "\n"
        "  3. NEVER write 'pending' as final status if process_order already ran and returned 'fulfilled' or 'partially_fulfilled'.\n"
        "\n"
        "  4. When summarizing order state, prefer ORDER_SNAPSHOT over conversation text.\n"
        "\n"
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

def _build_state_summary(state: RestaurantOrderState) -> str:
    """Deterministic, non-hallucinated summary derived from structured state."""
    lines: List[str] = []

    customer = state.get("customer")
    if isinstance(customer, dict) and customer:
        name = customer.get("name")
        allergies = customer.get("allergies") or []
        dietary = customer.get("dietary_restrictions") or []
        notes = customer.get("notes") or ""
        prefs = customer.get("preferences") or {}
        lines.append(
            "Customer profile: "
            + json.dumps(
                {
                    "name": name,
                    "allergies": allergies,
                    "dietary_restrictions": dietary,
                    "preferences": prefs,
                    "notes": notes,
                },
                ensure_ascii=False,
            )
        )

    resolved_id = _resolve_order_id_from_state(state, None)
    orders_state = cast(Dict[int, Dict[str, Any]], state.get("orders", {}))
    if resolved_id is None or resolved_id not in orders_state:
        lines.append("Active order: none")
        return "\n".join(lines).strip()

    order_data = orders_state[resolved_id]
    status = order_data.get("status")
    total_cost = order_data.get("total_cost")
    header = f"Active order: order_id={resolved_id} status={status}"
    if isinstance(total_cost, (int, float)) and total_cost:
        header += f" total_cost={float(total_cost):.2f}"
    lines.append(header)

    items = order_data.get("items") or []
    if not isinstance(items, list) or not items:
        lines.append("Items: none")
        return "\n".join(lines).strip()

    # Aggregate quantities by (status, name, special_instructions)
    agg: Dict[tuple[str, str, str], int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        qty = item.get("quantity")
        item_status = item.get("status")
        if not isinstance(name, str) or not isinstance(qty, int) or not isinstance(item_status, str):
            continue
        notes = _normalize_special_instructions(cast(Optional[str], item.get("special_instructions")))
        agg[(item_status, name, notes)] = agg.get((item_status, name, notes), 0) + int(qty)

    def _render_bucket(bucket_status: str) -> None:
        bucket_items = [(k, v) for (k, v) in agg.items() if k[0] == bucket_status]
        if not bucket_items:
            return
        lines.append(f"{bucket_status.title()} items:")
        for (st, name, notes), qty in sorted(bucket_items, key=lambda x: (x[0][1], x[0][2])):
            if notes:
                lines.append(f"- {qty} x {name} ({notes})")
            else:
                lines.append(f"- {qty} x {name}")

    _render_bucket("fulfilled")
    _render_bucket("pending")

    return "\n".join(lines).strip()


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
def cashier_calculate_total(
    order_id: int,
    *,
    state: Annotated[RestaurantOrderState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Calculate the total bill for an order including 15% tip.
    
    WHEN TO CALL:
    - User explicitly requests the bill: "check please", "can I have the bill"
    - User asks about total: "how much do I owe?", "what's my total?"
    - User indicates they're ready to pay: "I'll pay now", "ready to pay"
    
    DO NOT CALL:
    - Before user requests payment
    - If order is not yet fulfilled (check order status first)
    
    This updates order status to 'billed' and calculates: subtotal + 15% tip.
    
    Args:
        order_id: The order ID to calculate total for
        
    Returns:
        Total amount including tip, or error message if order not found
    """
    logger.info(f"Calculating total for order {order_id}")

    if not isinstance(state, dict):
        return Command(
            update={
                "messages": [ToolMessage(content="0.0", tool_call_id=tool_call_id)],
            }
        )

    orders_state = copy.deepcopy(cast(Dict[int, Dict[str, Any]], state.get("orders", {})))
    if not orders_state:
        logger.warning("No orders exist in the system")
        return Command(
            update={
                "messages": [ToolMessage(content="create an order first", tool_call_id=tool_call_id)]
            }
        )

    resolved_id = _resolve_order_id_from_state(state, order_id)
    if resolved_id is None or resolved_id not in orders_state:
        logger.error(f"Order {order_id} not found")
        return Command(
            update={"messages": [ToolMessage(content="0.0", tool_call_id=tool_call_id)]}
        )

    order_data = orders_state[resolved_id]
    # Enforce tool contract: do not bill until everything is fulfilled.
    if order_data.get("status") != "fulfilled":
        logger.warning(f"Cannot bill order {resolved_id} with status={order_data.get('status')}")
        return Command(
            update={"messages": [ToolMessage(content="order_not_fulfilled", tool_call_id=tool_call_id)]}
        )

    subtotal = 0.0
    for item in order_data["items"]:
        # Safety: only bill fulfilled items.
        if item.get("status") != "fulfilled":
            continue
        name = item["name"]
        qty = item["quantity"]
        price = menu_prices.get(name, 0.0)
        subtotal += price * qty

    tip_amount = subtotal * 0.15
    total = round(subtotal + tip_amount, 2)


    order_data["total_cost"] = total
    order_data["status"] = "billed"
    logger.info(f"Order {resolved_id} total calculated: ${total}")

    return Command(
        update={
            "orders": orders_state,
            "messages": [ToolMessage(content=str(total), tool_call_id=tool_call_id)],
        }
    )


@tool
def check_payment(
    amount: float,
    method: str,
    order_id: int,
    *,
    state: Annotated[RestaurantOrderState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Process customer payment for their order.
    
    WHEN TO CALL:
    - After cashier_calculate_total has been called and total communicated
    - User provides payment method and amount
    
    DO NOT CALL:
    - Before calculating and showing the total to customer
    - If order status is not 'billed'
    
    Payment methods:
    - "cash": Validates amount >= total, returns "cash_ok" with change or "insufficient_funds"
    - "card": Processes card (80% success rate, 20% "invalid" for testing)
    
    If payment fails:
    - Politely ask for alternative payment method
    - If second failure, offer manager assistance
    
    Args:
        amount: Payment amount provided by customer
        method: Payment method ("cash" or "card")
        order_id: The order ID to process payment for
        
    Returns:
        Payment status: "cash_ok", "insufficient_funds", "valid", "invalid", or error
    """
    logger.info(f"Processing payment: ${amount} via {method} for order {order_id}")

    if not isinstance(state, dict):
        return Command(update={"messages": [ToolMessage(content="order_not_found", tool_call_id=tool_call_id)]})

    orders_state = copy.deepcopy(cast(Dict[int, Dict[str, Any]], state.get("orders", {})))
    resolved_id = _resolve_order_id_from_state(state, order_id)
    if resolved_id is None or resolved_id not in orders_state:
        logger.error(f"Order {order_id} not found during payment")
        return Command(update={"messages": [ToolMessage(content="order_not_found", tool_call_id=tool_call_id)]})

    order_data = orders_state[resolved_id]

    # Ensure we have a billed total to pay against.
    total_due = float(order_data.get("total_cost") or 0.0)
    if total_due <= 0:
        subtotal = 0.0
        for item in order_data.get("items", []):
            name = item.get("name")
            qty = item.get("quantity")
            if isinstance(name, str) and isinstance(qty, int):
                subtotal += float(menu_prices.get(name, 0.0)) * float(qty)
        total_due = round(subtotal * 1.15, 2)
        order_data["total_cost"] = total_due
        order_data["status"] = "billed"

    result: str
    active_update: Optional[int] = cast(Optional[int], state.get("active_order_id"))

    if method.lower() == "cash":
        if amount < total_due:
            logger.warning(f"Insufficient cash payment: ${amount} < ${total_due}")
            result = "insufficient_funds"
        else:
            change = round(float(amount) - float(total_due), 2)
            order_data["status"] = "paid"
            active_update = None
            logger.info(f"Cash payment successful for order {resolved_id}. Change: ${change}")
            result = f"cash_ok with change {change}"

    elif method.lower() == "card":
        if random.random() < 0.8:
            order_data["status"] = "paid"
            active_update = None
            logger.info(f"Card payment successful for order {resolved_id}")
            result = "valid"
        else:
            logger.warning(f"Card payment failed for order {resolved_id}")
            result = "invalid"

    else:
        logger.error(f"Unknown payment method: {method}")
        result = "unknown_method"

    update: Dict[str, Any] = {
        "orders": orders_state,
        "messages": [ToolMessage(content=result, tool_call_id=tool_call_id)],
    }
    if active_update != cast(Optional[int], state.get("active_order_id")):
        update["active_order_id"] = active_update

    return Command(update=update)


@tool
def create_order(
    order_items: List[OrderItem],
    *,
    state: Annotated[RestaurantOrderState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Create a new order or update the active order with customer's items.
    
    WHEN TO CALL:
    - User confirms items they want to order
    - User modifies existing order (add/remove/change items)
    
    BEFORE CALLING:
    - Verify all item names match exact menu names (e.g., "steak" → clarify which: Filet Mignon or Ribeye Steak)
    - Convert colloquial terms: "iced tea" → "Black Tea", "coffee" → specific type
    
    AFTER CALLING:
    - Check the returned order status
    - If order created/updated, proceed to call process_order to fulfill it
    
    Format:
    [
        {"name": "Bruschetta", "quantity": 2},
        {"name": "Lobster Tail", "quantity": 1, "special_instructions": "no butter"}
    ]

    Quantity semantics:
    - quantity > 0: add items to the active order (or adjust totals in certain update scenarios)
    - quantity == 0: remove that item (by name + special_instructions) from the active order
    
    Args:
        order_items: List of items with name, quantity, and optional special_instructions
        
    Returns:
        Message with order ID and status (e.g., "Order 123 created with status 'pending'")
    """
    try:
        if not isinstance(state, dict):
            msg = "Error creating order: missing_state"
            return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

        orders_state = copy.deepcopy(cast(Dict[int, Dict[str, Any]], state.get("orders", {})))
        stocks_state = copy.deepcopy(cast(Dict[str, Dict[str, int]], state.get("department_stocks", {})))
        order_counter_state = int(state.get("order_counter", 0))

        incoming_quantities = _normalize_order_items(order_items)
        incoming_names = {name for (name, _) in incoming_quantities.keys()}
        incoming_keys = set(incoming_quantities.keys())

        active_id = _get_active_order_id_from_state(state)
        if active_id is None or active_id not in orders_state:
            logger.info(f"Creating new order with items: {order_items}")
            order_id = order_counter_state + 1
            orders_state[order_id] = {
                "items": [
                    {
                        "name": name,
                        "quantity": qty,
                        "special_instructions": special_instructions or "",
                        "department": "",
                        "status": "pending",
                    }
                    for (name, special_instructions), qty in incoming_quantities.items()
                    if int(qty) > 0
                ],
                "status": "pending",
                "total_cost": 0.0,
            }
            logger.info(f"Order {order_id} created successfully")
            msg = f"New order {order_id} created with status 'pending'."
            return Command(
                update={
                    "orders": orders_state,
                    "active_order_id": order_id,
                    "order_counter": order_id,
                    "department_stocks": stocks_state,
                    "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
                }
            )

        order_data = orders_state[active_id]
        existing_items = order_data.get("items", [])
        existing_names = {item["name"] for item in existing_items}
        overlap_ratio = 0.0
        if existing_names:
            overlap_ratio = len(existing_names & incoming_names) / len(existing_names)
        snapshot_mode = overlap_ratio >= 0.6
        logger.info(f"Updating order {active_id} (snapshot={snapshot_mode}) with items: {order_items}")

        if not snapshot_mode:
            new_items = list(existing_items)
            # Heuristic for modification requests like "make that two instead of one":
            # If the caller supplies exactly one item that already exists in the order,
            # interpret the provided quantity as the desired TOTAL for that item (not an additive delta).
            single_set_key: Optional[tuple[str, str]] = None
            single_set_qty: int = 0
            if len(incoming_quantities) == 1:
                (only_key, desired_qty) = next(iter(incoming_quantities.items()))
                only_name, only_instructions = only_key
                only_key_norm = _order_item_key(only_name, only_instructions)
                if any(
                    (
                        item.get("name") == only_name
                        and _order_item_key(
                            cast(str, item.get("name")),
                            cast(Optional[str], item.get("special_instructions")),
                        )
                        == only_key_norm
                    )
                    for item in existing_items
                ):
                    single_set_key = only_key_norm
                    single_set_qty = int(desired_qty)

            for (name, special_instructions), qty in incoming_quantities.items():
                key_norm = _order_item_key(name, special_instructions)
                if single_set_key is not None and key_norm == single_set_key:
                    matching_entries = [
                        item
                        for item in new_items
                        if item.get("name") == name
                        and _order_item_key(
                            cast(str, item.get("name")),
                            cast(Optional[str], item.get("special_instructions")),
                        )
                        == key_norm
                    ]
                    current_total = sum(int(item.get("quantity", 0)) for item in matching_entries)
                    raw_qty = int(single_set_qty)
                    # Try to tolerate both common calling conventions:
                    # - absolute: quantity == desired total
                    # - delta: quantity == amount to add (often "1 more")
                    if raw_qty == 0:
                        desired_total = 0
                    elif raw_qty == 1 and current_total > 0:
                        desired_total = current_total + 1
                    else:
                        desired_total = raw_qty

                    if desired_total > current_total:
                        delta = desired_total - current_total
                        if delta > 0:
                            new_items.append(
                                {
                                    "name": name,
                                    "quantity": delta,
                                    "special_instructions": special_instructions or "",
                                    "department": "",
                                    "status": "pending",
                                }
                            )
                        continue

                    remaining_to_remove = current_total - desired_total
                    if remaining_to_remove <= 0:
                        continue

                    adjusted_entries: List[Dict[str, object]] = []
                    # Remove pending first, then fulfilled (restocking any fulfilled quantities).
                    for entry in sorted(
                        matching_entries, key=lambda e: 0 if e.get("status") == "pending" else 1
                    ):
                        if remaining_to_remove <= 0:
                            adjusted_entries.append(entry)
                            continue
                        entry_qty = int(entry.get("quantity", 0))
                        if entry_qty <= remaining_to_remove:
                            if entry.get("status") == "fulfilled":
                                _restock_item_in_stocks(
                                    stocks_state,
                                    name,
                                    int(entry_qty),
                                    cast(str, entry.get("department", "")),
                                )
                            remaining_to_remove -= entry_qty
                            continue
                        if entry.get("status") == "fulfilled":
                            _restock_item_in_stocks(
                                stocks_state,
                                name,
                                int(remaining_to_remove),
                                cast(str, entry.get("department", "")),
                            )
                        updated_entry = dict(entry)
                        updated_entry["quantity"] = entry_qty - int(remaining_to_remove)
                        remaining_to_remove = 0
                        adjusted_entries.append(updated_entry)

                    # Replace all entries for this key with adjusted ones (keep other items unchanged).
                    new_items = [
                        item
                        for item in new_items
                        if not (
                            item.get("name") == name
                            and _order_item_key(
                                cast(str, item.get("name")),
                                cast(Optional[str], item.get("special_instructions")),
                            )
                            == key_norm
                        )
                    ] + adjusted_entries
                    continue

                if qty <= 0:
                    # For non-existent items, treat quantity==0 as a no-op in incremental mode.
                    continue
                pending_entry = next(
                    (
                        item
                        for item in new_items
                        if item["name"] == name
                        and _normalize_special_instructions(
                            cast(Optional[str], item.get("special_instructions"))
                        )
                        == _normalize_special_instructions(special_instructions)
                        and item["status"] == "pending"
                    ),
                    None,
                )
                if pending_entry:
                    pending_entry["quantity"] += qty
                else:
                    new_items.append(
                        {
                            "name": name,
                            "quantity": qty,
                            "special_instructions": special_instructions or "",
                            "department": "",
                            "status": "pending",
                        }
                    )
            order_data["items"] = new_items
        else:
            existing_by_key: Dict[tuple[str, str], List[Dict[str, object]]] = {}
            for item in existing_items:
                key = _order_item_key(
                    cast(str, item["name"]), cast(Optional[str], item.get("special_instructions"))
                )
                existing_by_key.setdefault(key, []).append(item)

            new_items: List[Dict[str, object]] = []
            for (name, special_instructions), desired_qty in incoming_quantities.items():
                key = _order_item_key(name, special_instructions)
                current_entries = existing_by_key.get(key, [])
                current_total = sum(item["quantity"] for item in current_entries)
                if not current_entries:
                    if desired_qty <= 0:
                        continue
                    new_items.append(
                        {
                            "name": name,
                            "quantity": desired_qty,
                            "special_instructions": special_instructions or "",
                            "department": "",
                            "status": "pending",
                        }
                    )
                    continue
                if desired_qty >= current_total:
                    new_items.extend(current_entries)
                    delta = desired_qty - current_total
                    if delta > 0:
                        new_items.append(
                            {
                                "name": name,
                                "quantity": delta,
                                "special_instructions": special_instructions or "",
                                "department": "",
                                "status": "pending",
                            }
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
                            _restock_item_in_stocks(
                                stocks_state, name, int(entry_qty), cast(str, entry.get("department", ""))
                            )
                        remaining_to_remove -= entry_qty
                        continue
                    if entry["status"] == "fulfilled":
                        _restock_item_in_stocks(
                            stocks_state, name, int(remaining_to_remove), cast(str, entry.get("department", ""))
                        )
                    updated_entry = dict(entry)
                    updated_entry["quantity"] = entry_qty - remaining_to_remove
                    remaining_to_remove = 0
                    adjusted_entries.append(updated_entry)
                new_items.extend(adjusted_entries)

            removed_keys = set(existing_by_key.keys()) - incoming_keys
            for removed_name, removed_instructions in removed_keys:
                for entry in existing_by_key.get((removed_name, removed_instructions), []):
                    if entry["status"] == "fulfilled":
                        _restock_item_in_stocks(
                            stocks_state,
                            cast(str, entry["name"]),
                            cast(int, entry["quantity"]),
                            cast(str, entry.get("department", "")),
                        )

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

        msg = f"Order {active_id} updated with status '{order_data['status']}'."
        return Command(
            update={
                "orders": orders_state,
                "department_stocks": stocks_state,
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    except ValidationError as e:
        logger.error(f"Order validation error: {str(e)}")
        msg = f"Error creating order: {str(e)}"
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})


@tool
def process_order(
    order_id: int,
    *,
    state: Annotated[RestaurantOrderState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Process an order by checking inventory and fulfilling items.
    
    WHEN TO CALL:
    - Immediately after create_order when order status is 'pending'
    - After create_order when order status is 'partially_fulfilled' (to retry pending items)
    
    BEFORE CALLING:
    - CHECK the current order status from previous create_order result
    - Verify status is 'pending' OR 'partially_fulfilled'
    
    DO NOT CALL:
    - If order status is 'fulfilled' (already done)
    - If order status is 'billed' or 'paid' (order complete)
    
    AFTER CALLING - CRITICAL:
    - READ the complete tool response JSON
    - If "ok": false → Explain the error, don't claim success
    - If "status": "insufficient_stock" → Tell customer exact available quantity, offer alternatives
    - If "status": "partially_fulfilled" → Explain which items succeeded and which failed
    - NEVER say items are "added" or "ordered" if tool reports insufficient stock
    
    Args:
        order_id: The order ID to process
        
    Returns:
        JSON with order status and per-item stock check results
    """
    logger.info(f"Processing order {order_id}")

    if not isinstance(state, dict):
        payload = json.dumps(
            {"tool": "process_order", "ok": False, "error": "missing_state", "order_id": order_id},
            ensure_ascii=False,
        )
        return Command(update={"messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)]})

    orders_state = copy.deepcopy(cast(Dict[int, Dict[str, Any]], state.get("orders", {})))
    stocks_state = copy.deepcopy(cast(Dict[str, Dict[str, int]], state.get("department_stocks", {})))

    resolved_id = _resolve_order_id_from_state(state, order_id)
    if resolved_id is None or resolved_id not in orders_state:
        logger.error(f"Order {order_id} not found")
        payload = json.dumps(
            {"tool": "process_order", "ok": False, "error": "order_not_found", "order_id": order_id},
            ensure_ascii=False,
        )
        return Command(update={"messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)]})

    order_data = orders_state[resolved_id]
    logger.info(f"Order {resolved_id} status: {order_data['status']}")
    if order_data["status"] not in ["pending", "partially_fulfilled"]:
        logger.warning(f"Cannot process order {resolved_id} in state: {order_data['status']}")
        payload = json.dumps(
            {
                "tool": "process_order",
                "ok": False,
                "error": f"cannot_process_order_in_state_{order_data['status']}",
                "order_id": resolved_id,
                "order_status": order_data["status"],
            },
            ensure_ascii=False,
        )
        return Command(update={"messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)]})

    all_fulfilled = True
    items_result: List[Dict[str, Any]] = []
    for item in order_data["items"]:
        if item["status"] == "pending":
            # call tool check_and_update_stock item_name, quantity
            logger.info(f"Checking stock for {item['name']} x{item['quantity']} in process_order")
            result = check_and_update_stock(stocks_state, cast(str, item["name"]), cast(int, item["quantity"]))
            logger.info(f"Stock check result for {item['name']}: {result} in process_order")
            if result.get("status") == "fulfilled":
                # Mark item fulfilled; record department
                dept = cast(str, result.get("department", ""))
                item["department"] = dept
                item["status"] = "fulfilled"
            elif result.get("status") == "insufficient_stock":
                all_fulfilled = False
            else:
                # item not found or another error
                all_fulfilled = False
            items_result.append(
                {
                    "name": item.get("name"),
                    "quantity": item.get("quantity"),
                    "special_instructions": item.get("special_instructions") or "",
                    "department": item.get("department") or "",
                    "item_status": item.get("status"),
                    "stock": result,
                }
            )
            continue

        items_result.append(
            {
                "name": item.get("name"),
                "quantity": item.get("quantity"),
                "special_instructions": item.get("special_instructions") or "",
                "department": item.get("department") or "",
                "item_status": item.get("status"),
                "stock": None,
            }
        )

    if all_fulfilled:
        order_data["status"] = "fulfilled"
        logger.info(f"Order {resolved_id} fully fulfilled")
    else:
        # If at least one item remains "pending", the order is partially fulfilled
        order_data["status"] = "partially_fulfilled"
        logger.warning(f"Order {resolved_id} partially fulfilled")

    payload = json.dumps(
        {
            "tool": "process_order",
            "ok": True,
            "order_id": resolved_id,
            "order_status": order_data["status"],
            "items": items_result,
        },
        ensure_ascii=False,
    )

    return Command(
        update={
            "orders": orders_state,
            "department_stocks": stocks_state,
            "messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)],
        }
    )


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
    update_customer_profile,
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

    # Inject structured, authoritative facts (so the model doesn't have to re-parse free text).
    customer = state.get("customer")
    if isinstance(customer, dict) and customer:
        model_messages.append(
            SystemMessage(
                content="Customer profile (structured, authoritative):\n"
                + json.dumps(customer, ensure_ascii=False)
            )
        )

    resolved_id = _resolve_order_id_from_state(state, None)
    orders_state = cast(Dict[int, Dict[str, Any]], state.get("orders", {}))
    if resolved_id is not None and resolved_id in orders_state:
        order_data = orders_state[resolved_id]
        order_snapshot = {
            "order_id": resolved_id,
            "status": order_data.get("status"),
            "total_cost": order_data.get("total_cost"),
            "items": order_data.get("items"),
        }
        model_messages.append(
            SystemMessage(
                content="Current active order (structured, authoritative):\n"
                + json.dumps(order_snapshot, ensure_ascii=False)
            )
        )

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

    # Summary should be accurate and non-hallucinated: derive it from structured state.
    try:
        new_summary = _build_state_summary(state)
    except Exception as e:
        logger.error(f"State summary generation failed: {str(e)}", exc_info=True)
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
workflow.add_node("init", _init_state_node)
workflow.add_node("agent", _agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("postprocess", _postprocess_node)

workflow.add_edge(START, "init")
workflow.add_edge("init", "agent")

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

def _reset_simulation_state() -> None:
    """Deprecated.

    This module now stores orders/stocks in LangGraph state (per thread_id), not globals.
    Kept only because golden-dataset runner calls it.
    """
    return None


def _load_golden_dataset_entries(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load the golden dataset JSON array and return usable dict entries."""
    raw = dataset_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Golden dataset must be a JSON array: {dataset_path}")
    entries: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("user_query"), str):
            entries.append(item)
    return entries


def _run_golden_dataset(dataset_path: Path) -> None:
    """Auto-run golden dataset queries through the agent graph for smoke testing."""
    if model_with_tools is None:
        raise RuntimeError("LLM is not initialized. Start the script via __main__.")

    entries = _load_golden_dataset_entries(dataset_path)
    if not entries:
        logger.warning(f"Golden dataset has no usable entries: {dataset_path}")
        return

    dataset_config = {"configurable": {"thread_id": "golden-dataset"}}
    print(f"\n[golden] Running {len(entries)} dataset queries from: {dataset_path.name}\n")
    logger.info(f"Golden dataset run start: {dataset_path} entries={len(entries)}")

    for fallback_idx, entry in enumerate(entries, 1):
        case_id = entry.get("id", fallback_idx)
        user_query = cast(str, entry["user_query"])

        logger.info(f"[golden #{case_id}] User query: {user_query}")
        print(f"[golden #{case_id}] You: {user_query}")

        user_message = HumanMessage(content=user_query, name="user")
        result = app.invoke({"messages": [user_message]}, config=dataset_config)
        ai_messages = [msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)]
        rendered = _render_ai_content(ai_messages[-1].content) if ai_messages else "<no response>"

        print(f"[golden #{case_id}] Waiter: {rendered}\n")
        logger.info(f"[golden #{case_id}] AI response: {rendered}")

    logger.info("Golden dataset run complete")
    print("[golden] Done.\n")

if __name__ == "__main__":
    logger.info("Starting restaurant waiter service with custom state")
    print("Welcome to the restaurant! Type 'q' to quit the conversation.")

    try:
        # Initialize model at runtime so missing env-vars don't print a full traceback on import.
        llm = _build_llm()
        model_with_tools = llm.bind_tools(tools)

        # Auto-run golden dataset on startup (if present) to smoke-test the agent framework.
        dataset_path = Path(__file__).with_name("waiter_agent_accessment_golden_dataset.json")
        if dataset_path.exists():
            try:
                _reset_simulation_state()
                _run_golden_dataset(dataset_path)
            except Exception as e:
                logger.error(f"Golden dataset run failed: {str(e)}", exc_info=True)
                print(f"[golden] Failed to run dataset: {e}")
            finally:
                # Ensure interactive mode starts from a clean slate.
                _reset_simulation_state()

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
