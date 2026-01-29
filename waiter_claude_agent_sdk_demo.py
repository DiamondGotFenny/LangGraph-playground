"""
Waiter Agent Demo using Claude Agent SDK

关于MCP使用说明：
本脚本使用的是 Claude Agent SDK 的内置MCP机制，而非外部MCP服务器。

- ✓ 使用了: SDK内置的 create_sdk_mcp_server() 将本地Python函数组织为MCP格式
- ✓ 目的: 利用MCP协议的工具调用规范，但工具实现都在本地Python代码中
- ✗ 没有使用: 外部MCP服务器（如文件系统MCP、数据库MCP等）
- ✗ 没有使用: 独立运行的MCP服务进程

简单来说：这是用MCP协议格式包装本地工具，而不是连接远程MCP服务。

工具重构说明：
本版本已将菜单查询工具进行了优化：
- 从3个独立工具（menu_food_list, menu_drinks_list, menu_price_get）
- 合并为1个统一工具（menu_query）
- 菜单数据整合了价格和分类信息
- 减少了50%的工具调用次数，提升了agent效率
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from logger_config import setup_logger
try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore
    _ = load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from claude_agent_sdk import (  # type: ignore
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
logger = logging.getLogger("VectorStoreAgentLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
CHAT_HISTORY: List[Tuple[str, str]] = []
MAX_HISTORY_ROUNDS = 5  # Maximum number of recent conversation rounds to keep in prompt

def conversation_reset() -> None:
    CHAT_HISTORY.clear()
def _configure_logging() -> str:
    """
    Configure logging to:
    - a stable file next to this script: waiter_claude_agent_sdk.log
    - an archival timestamped file under logs/
    """
    script_dir = Path(__file__).resolve().parent

    stable_log_path = script_dir / "waiter_claude_agent_sdk.log"

    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_log_path = log_dir / f"waiter_claude_agent_sdk_{ts}.log"

    global logger
    # Primary logger writes to the stable path (what people usually expect)
    logger = setup_logger(str(stable_log_path), log_to_console=False)

    # Also tee logs to a timestamped archival file (keeps per-run history)
    archive_fh = logging.FileHandler(str(archive_log_path), encoding="utf-8")
    archive_fh.setLevel(logging.INFO)
    archive_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(archive_fh)

    logger.info(f"Log file (stable): {stable_log_path}")
    logger.info(f"Log file (archive): {archive_log_path}")
    return str(stable_log_path)

def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None

def _normalize_tool_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw.startswith("mcp__waiter__"):
        return raw[len("mcp__waiter__"):]
    if raw.startswith("mcp__") and "__" in raw:
        return raw.split("__")[-1]
    return raw

def _extract_llm_token_usage_from_result(msg: ResultMessage) -> Optional[Dict[str, int]]:
    usage = getattr(msg, "usage", None)
    if not usage:
        return None
    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)

    prompt_tokens = _coerce_int(get("prompt_tokens"))
    completion_tokens = _coerce_int(get("completion_tokens"))
    total_tokens = _coerce_int(get("total_tokens"))

    cache_hit = _coerce_int(get("prompt_cache_hit_tokens"))
    cache_miss = _coerce_int(get("prompt_cache_miss_tokens"))

    input_tokens = _coerce_int(get("input_tokens"))
    output_tokens = _coerce_int(get("output_tokens"))
    cache_read = _coerce_int(get("cache_read_input_tokens"))
    cache_create = _coerce_int(get("cache_creation_input_tokens"))

    if prompt_tokens is None and input_tokens is not None:
        prompt_tokens = int(input_tokens) + int(cache_read or 0) + int(cache_create or 0)
    if completion_tokens is None and output_tokens is not None:
        completion_tokens = int(output_tokens)

    if cache_hit is None and cache_read is not None:
        cache_hit = int(cache_read)
    if cache_miss is None and cache_create is not None:
        cache_miss = int(cache_create)

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    if (
        prompt_tokens is None
        and completion_tokens is None
        and total_tokens is None
        and cache_hit is None
        and cache_miss is None
    ):
        return None

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0))),
        "cache_hit": int(cache_hit or 0),
        "cache_miss": int(cache_miss or 0),
    }

@dataclass
class ToolUsageTracker:
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    llm_responses: List[Dict[str, Any]] = field(default_factory=list)
    llm_token_totals: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }
    )
    llm_token_records: List[Dict[str, Any]] = field(default_factory=list)

    def record_tool_invocation(self, tool_name: str) -> None:
        name = _normalize_tool_name(tool_name)
        if not name:
            return
        self.tool_call_counts[name] = int(self.tool_call_counts.get(name, 0)) + 1

    def record_llm_response(self, tool_names: List[str]) -> None:
        normalized = [_normalize_tool_name(n) for n in tool_names if n]
        seen: Set[str] = set()
        unique_names: List[str] = []
        for n in normalized:
            if n in seen:
                continue
            seen.add(n)
            unique_names.append(n)
        self.llm_responses.append(
            {
                "index": len(self.llm_responses) + 1,
                "has_tools": bool(normalized),
                "tool_calls_count": len(normalized),
                "tool_names": unique_names,
            }
        )

    def record_llm_usage(self, usage: Optional[Dict[str, int]]) -> None:
        usage = usage or {}
        record = {
            "index": len(self.llm_token_records) + 1,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "cache_hit": int(usage.get("cache_hit") or 0),
            "cache_miss": int(usage.get("cache_miss") or 0),
        }
        self.llm_token_records.append(record)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cache_hit", "cache_miss"):
            self.llm_token_totals[k] = int(self.llm_token_totals.get(k) or 0) + int(record[k])

    def _render_tool_efficiency_table(self) -> str:
        total = sum(int(v) for v in self.tool_call_counts.values())
        lines: List[str] = []
        lines.append("工具调用效率统计")
        lines.append("")
        lines.append("| 工具名称 | 调用次数 | 占比 |")
        lines.append("| --- | ---: | ---: |")
        items = sorted(self.tool_call_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        for name, count in items:
            pct = (float(count) / float(total) * 100.0) if total else 0.0
            lines.append(f"| {name} | {int(count)} | {pct:.1f}% |")
        lines.append(f"| **总计** | **{int(total)}** | **100.0%** |" if total else "| **总计** | **0** | **0.0%** |")
        return "\n".join(lines)

    def _render_llm_tool_usage_table(self) -> str:
        lines: List[str] = []
        lines.append("LLM回答工具调用统计")
        lines.append("")
        lines.append("| 序号 | 是否调用工具 | 工具列表 | tool_calls数量 |")
        lines.append("| ---: | --- | --- | ---: |")
        for row in self.llm_responses:
            has_tools = "是" if row.get("has_tools") else "否"
            tools = row.get("tool_names") or []
            tools_str = ", ".join(tools) if isinstance(tools, list) else ""
            count = int(row.get("tool_calls_count") or 0)
            lines.append(f"| {int(row.get('index') or 0)} | {has_tools} | {tools_str} | {count} |")
        if not self.llm_responses:
            lines.append("| 0 | 否 |  | 0 |")
        return "\n".join(lines)

    def _render_llm_token_usage_table(self) -> str:
        lines: List[str] = []
        lines.append("LLM Token 使用统计")
        lines.append("")
        lines.append("| 序号 | prompt_tokens | completion_tokens | total_tokens | cache_hit | cache_miss |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        if not self.llm_token_records:
            lines.append("| 0 | 0 | 0 | 0 | 0 | 0 |")
        else:
            for row in self.llm_token_records:
                lines.append(
                    "| {idx} | {p} | {c} | {t} | {hit} | {miss} |".format(
                        idx=int(row.get("index") or 0),
                        p=int(row.get("prompt_tokens") or 0),
                        c=int(row.get("completion_tokens") or 0),
                        t=int(row.get("total_tokens") or 0),
                        hit=int(row.get("cache_hit") or 0),
                        miss=int(row.get("cache_miss") or 0),
                    )
                )
        totals = self.llm_token_totals or {}
        lines.append(
            "| **总计** | **{p}** | **{c}** | **{t}** | **{hit}** | **{miss}** |".format(
                p=int(totals.get("prompt_tokens") or 0),
                c=int(totals.get("completion_tokens") or 0),
                t=int(totals.get("total_tokens") or 0),
                hit=int(totals.get("cache_hit") or 0),
                miss=int(totals.get("cache_miss") or 0),
            )
        )
        return "\n".join(lines)

    def render_report(self) -> str:
        parts = [
            "",
            "====================",
            self._render_llm_token_usage_table(),
            "",
            self._render_tool_efficiency_table(),
            "",
            self._render_llm_tool_usage_table(),
            "====================",
            "",
        ]
        return "\n".join(parts)

_TOOL_USAGE = ToolUsageTracker()
# Menu structure with prices integrated
FOOD_MENU: Dict[str, Dict[str, Any]] = {
    # Appetizers
    "Bruschetta": {"price": 8.50, "category": "Appetizers"},
    "Caprese Salad": {"price": 9.00, "category": "Appetizers"},
    "Shrimp Cocktail": {"price": 12.50, "category": "Appetizers"},
    # Entrees
    "Salmon Fillet": {"price": 18.00, "category": "Entrees"},
    "Chicken Breast": {"price": 15.00, "category": "Entrees"},
    "Vegetable Stir-Fry": {"price": 14.00, "category": "Entrees"},
    # Main Courses
    "Filet Mignon": {"price": 35.00, "category": "Main Courses"},
    "Lobster Tail": {"price": 40.00, "category": "Main Courses"},
    "Rack of Lamb": {"price": 32.00, "category": "Main Courses"},
    # Side Dishes
    "Mashed Potatoes": {"price": 5.00, "category": "Side Dishes"},
    "Grilled Asparagus": {"price": 6.00, "category": "Side Dishes"},
    "Roasted Vegetables": {"price": 5.50, "category": "Side Dishes"},
    # Desserts
    "Chocolate Cake": {"price": 8.00, "category": "Desserts"},
    "Cheesecake": {"price": 7.50, "category": "Desserts"},
    "Tiramisu": {"price": 9.00, "category": "Desserts"},
    # Pasta
    "Spaghetti Carbonara": {"price": 16.00, "category": "Pasta"},
    "Fettuccine Alfredo": {"price": 17.00, "category": "Pasta"},
    "Lasagna": {"price": 19.00, "category": "Pasta"},
    # Grill Station
    "Ribeye Steak": {"price": 30.00, "category": "Grill Station"},
    "BBQ Ribs": {"price": 25.00, "category": "Grill Station"},
    "Grilled Salmon": {"price": 28.00, "category": "Grill Station"},
    # Cold Station
    "Caesar Salad": {"price": 10.00, "category": "Cold Station"},
    "Greek Salad": {"price": 9.50, "category": "Cold Station"},
    "Fruit Platter": {"price": 12.00, "category": "Cold Station"},
}

DRINKS_MENU: Dict[str, Dict[str, Any]] = {
    # Alcoholic
    "Red Wine": {"price": 9.00, "category": "Alcoholic"},
    "White Wine": {"price": 9.00, "category": "Alcoholic"},
    "Cocktail": {"price": 12.00, "category": "Alcoholic"},
    "Beer": {"price": 7.00, "category": "Alcoholic"},
    # Coffee & Tea
    "Espresso": {"price": 3.00, "category": "Coffee & Tea"},
    "Cappuccino": {"price": 4.50, "category": "Coffee & Tea"},
    "Latte": {"price": 4.50, "category": "Coffee & Tea"},
    "Green Tea": {"price": 3.50, "category": "Coffee & Tea"},
    "Black Tea": {"price": 3.50, "category": "Coffee & Tea"},
}

# Keep a flat price lookup for internal use (billing)
MENU_PRICES: Dict[str, float] = {}
MENU_PRICES.update({name: info["price"] for name, info in FOOD_MENU.items()})
MENU_PRICES.update({name: info["price"] for name, info in DRINKS_MENU.items()})
INITIAL_STOCKS: Dict[str, Dict[str, int]] = {
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
def _render_menu(menu: Dict[str, Dict[str, Any]], title: str) -> str:
    """Render menu with prices grouped by category."""
    categories: Dict[str, List[Tuple[str, float]]] = {}
    for name, info in menu.items():
        category = info.get("category", "Other")
        price = info.get("price", 0.0)
        if category not in categories:
            categories[category] = []
        categories[category].append((name, price))
    
    lines = [title]
    for category in sorted(categories.keys()):
        lines.append(f"({category})")
        for name, price in sorted(categories[category]):
            lines.append(f"- {name}: ${price:.2f}")
        lines.append("")
    
    return "\n".join(lines).strip()
RESTAURANT_INFO_TEXT = "\n".join([
   """
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
])
StateT = Dict[str, Any]
STATE: StateT = {}
def _normalize_notes(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())
def _build_alias_map(items: Iterable[str]) -> Dict[str, str]:
    mapping = {s.lower(): s for s in items}
    mapping.update({"iced tea": "Black Tea", "ice tea": "Black Tea", "veggies": "Roasted Vegetables", "vegetables": "Roasted Vegetables", "veg": "Roasted Vegetables"})
    return mapping
ALIASES = _build_alias_map(MENU_PRICES.keys())
def _resolve_item(name: str) -> Optional[str]:
    key = str(name or "").strip().lower()
    return ALIASES.get(key) if key else None
def _find_department(stocks: Dict[str, Dict[str, int]], item: str) -> Optional[str]:
    for dept, items in stocks.items():
        if item in items:
            return dept
    return None
def state_reset() -> None:
    STATE.clear()
    STATE["order_counter"] = 0
    STATE["active_order_id"] = None
    STATE["orders"] = {}
    STATE["stocks"] = {dept: dict(items) for dept, items in INITIAL_STOCKS.items()}
    STATE["note"] = {}
    STATE["summary"] = ""  # Summary of older conversation history beyond the sliding window
def active_order() -> Optional[StateT]:
    order_id = STATE.get("active_order_id")
    if order_id is None:
        return None
    return STATE["orders"].get(int(order_id))
def _note() -> Dict[Tuple[str, str], int]:
    return STATE["note"]
def _render_note() -> str:
    note = _note()
    if not note:
        return "Note (draft): empty"
    lines = ["Note (draft):"]
    for (name, notes), qty in sorted(note.items(), key=lambda x: (x[0][0], x[0][1])):
        q = int(qty)
        if q <= 0:
            continue
        lines.append(f"- {q} x {name}" + (f" ({notes})" if notes else ""))
    return "\n".join(lines).strip()
def _render_order(order: Optional[StateT]) -> str:
    if not order:
        return "Active order: none"
    header = f"Active order: order_id={order['order_id']} status={order.get('status')}"
    if order.get("total_cost"):
        header += f" total_cost={float(order['total_cost']):.2f}"
    lines = [header]
    items: List[StateT] = list(order.get("items") or [])
    if not items:
        lines.append("Items: none")
        return "\n".join(lines).strip()
    for status in ("fulfilled", "pending"):
        agg: Dict[Tuple[str, str], int] = {}
        for it in items:
            if it.get("status") != status:
                continue
            key = (str(it.get("name") or ""), _normalize_notes(str(it.get("notes") or "")))
            agg[key] = agg.get(key, 0) + int(it.get("quantity") or 0)
        if not agg:
            continue
        lines.append(f"{status.title()} items:")
        for (iname, inotes), iqty in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
            lines.append(f"- {iqty} x {iname}" + (f" ({inotes})" if inotes else ""))
    return "\n".join(lines).strip()
def _model_context() -> str:
    return "\n".join([_render_order(active_order()), _render_note()]).strip()
def note_add(name: str, quantity: int, notes: str) -> Tuple[bool, str]:
    item = _resolve_item(name)
    if not item:
        return False, "Unknown menu item. Use menu tools to verify the exact name."
    qty = int(quantity)
    if qty <= 0:
        return False, "Quantity must be > 0."
    key = (item, _normalize_notes(notes))
    note = _note()
    note[key] = int(note.get(key, 0)) + qty
    return True, _render_note()
def note_set_quantity(name: str, quantity: int, notes: str) -> Tuple[bool, str]:
    item = _resolve_item(name)
    if not item:
        return False, "Unknown menu item. Use menu tools to verify the exact name."
    qty = int(quantity)
    if qty < 0:
        return False, "Quantity must be >= 0."
    key = (item, _normalize_notes(notes))
    note = _note()
    if qty == 0:
        note.pop(key, None)
    else:
        note[key] = qty
    return True, _render_note()
def note_clear() -> str:
    STATE["note"] = {}
    return _render_note()
def note_load_from_order(order_id: int) -> Tuple[bool, str]:
    oid = int(order_id)
    order = active_order() if oid <= 0 else STATE["orders"].get(oid)
    if not order:
        return False, "No such order to load."
    agg: Dict[Tuple[str, str], int] = {}
    for it in order.get("items") or []:
        key = (str(it.get("name") or ""), _normalize_notes(str(it.get("notes") or "")))
        agg[key] = agg.get(key, 0) + int(it.get("quantity") or 0)
    STATE["note"] = agg
    return True, _render_note()
def order_create_from_note() -> Tuple[bool, str]:
    if not _note():
        return False, "Note is empty."
    a = active_order()
    if a and a.get("status") != "paid":
        return False, f"Active order exists (order_id={a['order_id']}). Use update tool."
    STATE["order_counter"] = int(STATE["order_counter"]) + 1
    oid = int(STATE["order_counter"])
    order: StateT = {"order_id": oid, "status": "created", "total_cost": None, "items": []}
    stocks = STATE["stocks"]
    for (iname, inotes), iqty in sorted(_note().items(), key=lambda x: (x[0][0], x[0][1])):
        q = int(iqty)
        if q <= 0:
            continue
        dept = _find_department(stocks, iname) or ""
        order["items"].append({"name": iname, "quantity": q, "notes": inotes, "status": "pending", "department": dept})
    STATE["orders"][oid] = order
    STATE["active_order_id"] = oid
    STATE["note"] = {}
    return True, json.dumps({"tool": "order_create_from_note", "ok": True, "order_id": oid}, ensure_ascii=False)
def _restock(name: str, qty: int, dept: str) -> None:
    if qty <= 0:
        return
    stock = STATE["stocks"].get(dept)
    if not stock or name not in stock:
        return
    stock[name] = int(stock[name]) + int(qty)
def _remove_from_order(order: StateT, name: str, notes: str, qty: int) -> int:
    to_remove = int(qty)
    if to_remove <= 0:
        return 0
    notes_norm = _normalize_notes(notes)
    def match(it: StateT) -> bool:
        if it.get("name") != name:
            return False
        if notes_norm:
            return _normalize_notes(str(it.get("notes") or "")) == notes_norm
        return True
    removed = 0
    items: List[StateT] = list(order.get("items") or [])
    for it in list(items):
        if to_remove <= 0:
            break
        if it.get("status") != "pending" or not match(it):
            continue
        take = min(int(it.get("quantity") or 0), to_remove)
        it["quantity"] = int(it.get("quantity") or 0) - take
        to_remove -= take
        removed += take
        if int(it["quantity"]) <= 0:
            items.remove(it)
    for it in list(items):
        if to_remove <= 0:
            break
        if it.get("status") != "fulfilled" or not match(it):
            continue
        take = min(int(it.get("quantity") or 0), to_remove)
        it["quantity"] = int(it.get("quantity") or 0) - take
        to_remove -= take
        removed += take
        _restock(name, take, str(it.get("department") or ""))
        if int(it["quantity"]) <= 0:
            items.remove(it)
    order["items"] = items
    return removed
def _add_pending(order: StateT, name: str, notes: str, qty: int) -> None:
    q = int(qty)
    if q <= 0:
        return
    dept = _find_department(STATE["stocks"], name) or ""
    order["items"].append({"name": name, "quantity": q, "notes": notes, "status": "pending", "department": dept})
def order_update_to_match_note(order_id: int) -> Tuple[bool, str]:
    oid = int(order_id)
    order = active_order() if oid <= 0 else STATE["orders"].get(oid)
    if not order:
        return False, "No active order to update."
    if order.get("status") == "paid":
        return False, "Order is already paid; cannot update."
    if not _note():
        return False, "Note is empty."
    current: Dict[Tuple[str, str], int] = {}
    for it in order.get("items") or []:
        key = (str(it.get("name") or ""), _normalize_notes(str(it.get("notes") or "")))
        current[key] = current.get(key, 0) + int(it.get("quantity") or 0)
    target = dict(_note())
    changes: List[StateT] = []
    for key in sorted(set(current.keys()) | set(target.keys()), key=lambda x: (x[0], x[1])):
        cur = int(current.get(key, 0))
        tgt = int(target.get(key, 0))
        iname, inotes = key
        if cur == tgt:
            continue
        if tgt > cur:
            delta = tgt - cur
            _add_pending(order, iname, inotes, delta)
            changes.append({"action": "add", "name": iname, "notes": inotes, "quantity": delta})
        else:
            delta = cur - tgt
            removed = _remove_from_order(order, iname, inotes, delta)
            changes.append({"action": "remove", "name": iname, "notes": inotes, "quantity": removed})
    STATE["note"] = {}
    return True, json.dumps({"tool": "order_update_to_match_note", "ok": True, "order_id": int(order["order_id"]), "changes": changes}, ensure_ascii=False)
def inventory_check(name: str, quantity: int) -> Tuple[bool, str]:
    item = _resolve_item(name)
    if not item:
        return False, "Unknown menu item."
    q = int(quantity)
    if q <= 0:
        return False, "Quantity must be > 0."
    dept = _find_department(STATE["stocks"], item)
    if not dept:
        return False, "Item not found in inventory."
    available = int(STATE["stocks"][dept].get(item, 0))
    status = "fulfilled" if available >= q else "insufficient_stock"
    return True, json.dumps({"tool": "inventory_check", "ok": True, "name": item, "requested": q, "available": available, "status": status}, ensure_ascii=False)
def order_process(order_id: int) -> Tuple[bool, str]:
    oid = int(order_id)
    order = active_order() if oid <= 0 else STATE["orders"].get(oid)
    if not order:
        return False, "No active order to process."
    if order.get("status") == "paid":
        return False, "Order is already paid."
    stocks = STATE["stocks"]
    results: List[StateT] = []
    any_pending = False
    for it in list(order.get("items") or []):
        if it.get("status") != "pending":
            continue
        any_pending = True
        iname = str(it.get("name") or "")
        qty = int(it.get("quantity") or 0)
        dept = str(it.get("department") or "") or (_find_department(stocks, iname) or "")
        it["department"] = dept
        if not dept:
            results.append({"name": iname, "quantity": qty, "item_status": "item_not_found"})
            continue
        available = int(stocks[dept].get(iname, 0))
        if available <= 0:
            results.append({"name": iname, "quantity": qty, "item_status": "pending", "stock": {"available": 0}})
            continue
        if available >= qty:
            stocks[dept][iname] = available - qty
            it["status"] = "fulfilled"
            results.append({"name": iname, "quantity": qty, "item_status": "fulfilled"})
            continue
        fulfilled_qty = available
        remaining = qty - available
        stocks[dept][iname] = 0
        it["quantity"] = remaining
        order["items"].append({"name": iname, "quantity": fulfilled_qty, "notes": str(it.get("notes") or ""), "status": "fulfilled", "department": dept})
        results.append({"name": iname, "quantity": qty, "item_status": "partially_fulfilled", "stock": {"available": fulfilled_qty}})
    if not any_pending:
        return True, json.dumps({"tool": "order_process", "ok": True, "order_id": int(order["order_id"]), "items": []}, ensure_ascii=False)
    order["status"] = "partially_fulfilled" if any(i.get("status") == "pending" for i in order["items"]) else "fulfilled"
    return True, json.dumps({"tool": "order_process", "ok": True, "order_id": int(order["order_id"]), "order_status": order["status"], "items": results}, ensure_ascii=False)
def bill_calculate_total(order_id: int) -> Tuple[bool, str]:
    oid = int(order_id)
    order = active_order() if oid <= 0 else STATE["orders"].get(oid)
    if not order:
        return False, "No active order to bill."
    if order.get("status") != "fulfilled":
        return False, "order_not_fulfilled"
    subtotal = 0.0
    for it in order.get("items") or []:
        if it.get("status") != "fulfilled":
            continue
        subtotal += float(MENU_PRICES.get(str(it.get("name") or ""), 0.0)) * float(int(it.get("quantity") or 0))
    total = round(subtotal * 1.15, 2)
    order["total_cost"] = total
    order["status"] = "billed"
    return True, json.dumps({"tool": "bill_calculate_total", "ok": True, "order_id": int(order["order_id"]), "subtotal": round(subtotal, 2), "total": total}, ensure_ascii=False)
def payment_charge(order_id: int, method: str, amount: float) -> Tuple[bool, str]:
    oid = int(order_id)
    order = active_order() if oid <= 0 else STATE["orders"].get(oid)
    if not order:
        return False, "order_not_found"
    total_due = float(order.get("total_cost") or 0.0)
    if total_due <= 0:
        return False, "order_not_billed"
    m = str(method or "").strip().lower()
    amt = float(amount or 0.0)
    if m == "cash":
        if amt < total_due:
            status = "insufficient_funds"
        else:
            status = f"cash_ok with change {round(amt - total_due, 2)}"
            order["status"] = "paid"
            STATE["active_order_id"] = None
    elif m == "card":
        if random.random() < 0.8:
            status = "valid"
            order["status"] = "paid"
            STATE["active_order_id"] = None
        else:
            status = "invalid"
    else:
        return False, "unsupported_payment_method"
    return True, json.dumps({"tool": "payment_charge", "ok": True, "order_id": int(order["order_id"]), "method": m, "amount": amt, "total_due": total_due, "status": status}, ensure_ascii=False)
def _tool_text(text: str, *, is_error: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"content": [{"type": "text", "text": str(text)}]}
    if is_error:
        payload["is_error"] = True
    return payload
@tool("restaurant_info_get", "Get restaurant info (history, awards).", {})
async def restaurant_info_get(_args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("tool_call: restaurant_info_get args={}")
    _TOOL_USAGE.record_tool_invocation("restaurant_info_get")
    return _tool_text(RESTAURANT_INFO_TEXT)

@tool(
    "menu_query",
    "Query menu: list all food/drinks with prices, or check if item exists and get its price.",
    {"menu_type": str, "item_name": str}
)
async def menu_query(args: Dict[str, Any]) -> Dict[str, Any]:
    menu_type = str(args.get("menu_type") or "").strip().lower()
    item_name = str(args.get("item_name") or "").strip()
    
    logger.info(f"tool_call: menu_query args={json.dumps({'menu_type': menu_type, 'item_name': item_name}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("menu_query")
    
    # If item_name is provided, query specific item
    if item_name:
        resolved = _resolve_item(item_name)
        if not resolved:
            return _tool_text(f"Item '{item_name}' not found in menu.", is_error=True)
        
        # Check which menu it belongs to
        if resolved in FOOD_MENU:
            info = FOOD_MENU[resolved]
            result = f"{resolved}: ${info['price']:.2f} (Category: {info['category']}, Type: Food)"
        elif resolved in DRINKS_MENU:
            info = DRINKS_MENU[resolved]
            result = f"{resolved}: ${info['price']:.2f} (Category: {info['category']}, Type: Drink)"
        else:
            return _tool_text(f"Item '{resolved}' not found in menu.", is_error=True)
        
        return _tool_text(result)
    
    # List full menu based on menu_type
    if menu_type in ("food", "foods"):
        return _tool_text(_render_menu(FOOD_MENU, "Food Menu:"))
    elif menu_type in ("drink", "drinks", "beverage", "beverages"):
        return _tool_text(_render_menu(DRINKS_MENU, "Drinks Menu:"))
    elif menu_type in ("all", "both", ""):
        food_text = _render_menu(FOOD_MENU, "Food Menu:")
        drinks_text = _render_menu(DRINKS_MENU, "Drinks Menu:")
        return _tool_text(f"{food_text}\n\n{drinks_text}")
    else:
        return _tool_text(
            "Invalid menu_type. Use 'food' for food menu, 'drinks' for drinks menu, or 'all' for both. "
            "To check a specific item, provide item_name.",
            is_error=True
        )
@tool("inventory_check", "Check stock for an item (does not reserve).", {"name": str, "quantity": int})
async def inventory_check_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name") or "")
    quantity = int(args.get("quantity") or 0)
    logger.info(f"tool_call: inventory_check args={json.dumps({'name': name, 'quantity': quantity}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("inventory_check")
    ok, out = inventory_check(name, quantity)
    return _tool_text(out, is_error=not ok)
@tool("note_view", "Show active order + note (draft).", {})
async def note_view(_args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("tool_call: note_view args={}")
    _TOOL_USAGE.record_tool_invocation("note_view")
    return _tool_text(_model_context())
@tool("note_clear", "Clear the note (draft).", {})
async def note_clear_tool(_args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("tool_call: note_clear args={}")
    _TOOL_USAGE.record_tool_invocation("note_clear")
    return _tool_text(note_clear())
@tool("note_load_from_order", "Load an order into the note (pass 0 to load active order).", {"order_id": int})
async def note_load_from_order_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = int(args.get("order_id") or 0)
    logger.info(f"tool_call: note_load_from_order args={json.dumps({'order_id': order_id}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("note_load_from_order")
    ok, out = note_load_from_order(order_id)
    return _tool_text(out, is_error=not ok)
@tool("note_add_item", "Add an item to the note (draft).", {"name": str, "quantity": int, "notes": str})
async def note_add_item_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name") or "")
    quantity = int(args.get("quantity") or 0)
    notes = str(args.get("notes") or "")
    logger.info(f"tool_call: note_add_item args={json.dumps({'name': name, 'quantity': quantity, 'notes': notes}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("note_add_item")
    ok, out = note_add(name, quantity, notes)
    return _tool_text(out, is_error=not ok)
@tool("note_set_item_quantity", "Set an item's quantity in the note (0 removes).", {"name": str, "quantity": int, "notes": str})
async def note_set_item_quantity_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name") or "")
    quantity = int(args.get("quantity") or 0)
    notes = str(args.get("notes") or "")
    logger.info(f"tool_call: note_set_item_quantity args={json.dumps({'name': name, 'quantity': quantity, 'notes': notes}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("note_set_item_quantity")
    ok, out = note_set_quantity(name, quantity, notes)
    return _tool_text(out, is_error=not ok)
@tool("order_create_from_note", "Create a new order from the note (clears the note).", {})
async def order_create_from_note_tool(_args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("tool_call: order_create_from_note args={}")
    _TOOL_USAGE.record_tool_invocation("order_create_from_note")
    ok, out = order_create_from_note()
    return _tool_text(out, is_error=not ok)
@tool("order_update_to_match_note", "Update an order to match the note (clears the note). Pass 0 to update active order.", {"order_id": int})
async def order_update_to_match_note_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = int(args.get("order_id") or 0)
    logger.info(f"tool_call: order_update_to_match_note args={json.dumps({'order_id': order_id}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("order_update_to_match_note")
    ok, out = order_update_to_match_note(order_id)
    return _tool_text(out, is_error=not ok)
@tool("order_view", "Show the active order summary.", {})
async def order_view_tool(_args: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("tool_call: order_view args={}")
    _TOOL_USAGE.record_tool_invocation("order_view")
    return _tool_text(_render_order(active_order()))
@tool("order_process", "Process the order: fulfill pending items and report stock issues. Pass 0 for active order.", {"order_id": int})
async def order_process_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = int(args.get("order_id") or 0)
    logger.info(f"tool_call: order_process args={json.dumps({'order_id': order_id}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("order_process")
    ok, out = order_process(order_id)
    return _tool_text(out, is_error=not ok)
@tool("bill_calculate_total", "Calculate the check (15% tip). Requires fulfilled order. Pass 0 for active order.", {"order_id": int})
async def bill_calculate_total_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = int(args.get("order_id") or 0)
    logger.info(f"tool_call: bill_calculate_total args={json.dumps({'order_id': order_id}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("bill_calculate_total")
    ok, out = bill_calculate_total(order_id)
    return _tool_text(out, is_error=not ok)
@tool("payment_charge", "Charge payment (cash/card) for a billed order. Pass 0 for active order.", {"order_id": int, "method": str, "amount": float})
async def payment_charge_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = int(args.get("order_id") or 0)
    method = str(args.get("method") or "")
    amount = float(args.get("amount") or 0.0)
    logger.info(f"tool_call: payment_charge args={json.dumps({'order_id': order_id, 'method': method, 'amount': amount}, ensure_ascii=False)}")
    _TOOL_USAGE.record_tool_invocation("payment_charge")
    ok, out = payment_charge(order_id, method, amount)
    return _tool_text(out, is_error=not ok)
SERVER = create_sdk_mcp_server(name="waiter", version="1.0.0", tools=[restaurant_info_get, menu_query, inventory_check_tool, note_view, note_clear_tool, note_load_from_order_tool, note_add_item_tool, note_set_item_quantity_tool, order_create_from_note_tool, order_update_to_match_note_tool, order_view_tool, order_process_tool, bill_calculate_total_tool, payment_charge_tool])
SYSTEM_PROMPT = "\n".join([
    "You are a professional restaurant waiter at Villa Toscana (upscale Italian).",
    "",
    "Serve efficiently and accurately.",
    "",
    "=== HARD CONSTRAINTS (MUST FOLLOW) ===",
    "",
    "1. NO HALLUCINATION: NEVER mention, suggest, or recommend any dish/drink that you have not verified exists in the menu.",
    "   - Before recommending dishes to customer, you MUST first call menu_query to retrieve the actual menu.",
    "   - If customer asks 'what do you recommend?', query the menu FIRST, then recommend from actual results.",
    "   - ONLY mention items you have confirmed exist via menu_query in the current conversation.",
    "",
    "2. ORDER CONFIRMATION REQUIRED: NEVER create/commit an order until the customer has finished ordering AND confirmed.",
    "   ",
    "   Be aware: customers often order multiple items across several turns. Your job is to sense whether they are:",
    "   (a) Still in the middle of ordering — they may add more items, so just acknowledge and ask 'Anything else?'",
    "   (b) Done ordering — they signal completion, so confirm the full order and ask if you should place it.",
    "   ",
    "   Signs that customer is STILL ORDERING (do NOT ask to place order yet):",
    "   - They just named one or two items without closing language",
    "   - They say 'and also...', 'plus...', 'let me think...', or pause mid-thought",
    "   - The conversation flow suggests they're browsing or deciding",
    "   ",
    "   Signs that customer has FINISHED ordering (now confirm before placing):",
    "   - Explicit closure: 'that's all', 'nothing else', 'I'm done', 'that's it for now'",
    "   - They ask about the total, or repeat/confirm what they ordered",
    "   - They say 'go ahead', 'place it', 'confirm', 'yes, that's everything'",
    "   ",
    "   When you sense they're done, summarize the order and ask: 'Shall I place this order for you?'",
    "   Only call order_create_from_note or order_update_to_match_note AFTER they confirm yes.",
    "   ",
    "   Key: Don't interrupt the ordering flow with premature confirmations. Be patient and attentive.",
    "",
    "=== WORKFLOW ===",
    "",
    "1) Understand the customer; ask 1-2 clarifying questions only when needed (e.g., 'steak' is ambiguous).",
    "",
    "2) Use menu_query tool efficiently:",
    "   - To list food menu with prices: menu_query(menu_type='food')",
    "   - To list drinks menu with prices: menu_query(menu_type='drinks')",
    "   - To check if item exists and get its price: menu_query(menu_type='', item_name='item')",
    "   - One call can verify item existence, category, and price simultaneously.",
    "",
    "3) Use inventory_check for stock availability.",
    "",
    "4) Use NOTE (draft) before committing: update the note to reflect the customer's desired items.",
    "   Do NOT say the order is placed until you commit.",
    "",
    "5) CONFIRM with customer before committing. Then:",
    "   - If no active order: order_create_from_note",
    "   - If order exists: order_update_to_match_note (never create duplicates)",
    "   - After committing: order_process to fulfill pending items and surface stock issues.",
    "",
    "6) Billing/payment: only bill when asked; then payment_charge; if card invalid, ask politely for another method.",
    "",
    "=== STYLE ===",
    "",
    "Friendly, concise, proactive.",
    "",
    "Do NOT repeatedly list the full set of items the customer has ordered in every response.",
    " - When the customer adds something, briefly acknowledge the new item (e.g., 'Sure, a Margherita pizza for you.')",
    " - Save full, structured recaps of the entire order for:",
    "   (a) When you sense the customer has finished ordering and you are asking for final confirmation, or",
    "   (b) When the customer explicitly asks you to repeat/confirm everything they ordered.",
    "",
    "Don't spam tools; reuse verified info already in conversation when safe.",
    "Each turn includes a [CONTEXT] block showing Active order + Note; treat it as authoritative internal state.",
])
async def _summarize_old_history(old_rounds: List[Tuple[str, str]], options: ClaudeAgentOptions) -> str:
    """Summarize older conversation rounds that will be truncated from the sliding window."""
    if not old_rounds:
        return ""
    
    # Build a prompt to summarize the old conversation
    conversation_text = []
    for user_text, assistant_text in old_rounds:
        conversation_text.append(f"Customer: {user_text}")
        conversation_text.append(f"Waiter: {assistant_text}")
    
    summary_prompt = f"""Please provide a brief summary of the following conversation between a customer and waiter at Villa Toscana restaurant. Focus on:
- What the customer ordered or discussed
- Any important preferences or special requests
- Current status of orders or interactions

Conversation to summarize:
{chr(10).join(conversation_text)}

Provide a concise summary (2-3 sentences max):"""
    
    # Create a temporary client to get the summary (without MCP servers to avoid recursion)
    temp_options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant that summarizes conversations.",
        include_partial_messages=False
    )
    if hasattr(options, 'model') and options.model:
        temp_options.model = options.model
    
    async with ClaudeSDKClient(options=temp_options) as client:
        await client.query(summary_prompt)
        parts: List[str] = []
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        summary = "".join(parts).strip()
    
    logger.info(f"Generated summary for {len(old_rounds)} old rounds: {summary}")
    return summary

def _render_history() -> str:
    if not CHAT_HISTORY:
        return ""
    tail = CHAT_HISTORY[-MAX_HISTORY_ROUNDS:]
    lines: List[str] = ["[HISTORY]"]
    for user_text, assistant_text in tail:
        lines.append(f"Customer: {user_text}")
        lines.append(f"Waiter: {assistant_text}")
    lines.append("[/HISTORY]")
    return "\n".join(lines).strip()
def _build_prompt(user_text: str) -> str:
    summary = STATE.get("summary", "").strip()
    summary_block = f"[SUMMARY OF EARLIER CONVERSATION]\n{summary}\n[/SUMMARY]\n\n" if summary else ""
    
    history = _render_history()
    history_block = f"{history}\n\n" if history else ""
    
    return f"[CONTEXT]\n{_model_context()}\n[/CONTEXT]\n\n{summary_block}{history_block}Customer: {user_text}\n"
async def _drain_response(client: ClaudeSDKClient) -> Tuple[str, Optional[Dict[str, int]], List[str]]:
    parts: List[str] = []
    tool_names: List[str] = []
    usage: Optional[Dict[str, int]] = None
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    logger.info(f"tool_use: {block.name} input={json.dumps(block.input, ensure_ascii=False)}")
                    if getattr(block, "name", None):
                        tool_names.append(str(block.name))
        elif isinstance(msg, ResultMessage):
            if msg.total_cost_usd:
                logger.info(f"Cost USD: {msg.total_cost_usd:.6f}")
            extracted = _extract_llm_token_usage_from_result(msg)
            if extracted:
                usage = extracted
    rendered = "".join(parts).strip()
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered, usage, tool_names
async def _run_turn(user_text: str, options: ClaudeAgentOptions) -> str:
    prompt = _build_prompt(user_text)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        rendered, usage, tool_names = await _drain_response(client)
    _TOOL_USAGE.record_llm_response(tool_names)
    _TOOL_USAGE.record_llm_usage(usage)
    CHAT_HISTORY.append((user_text, rendered))
    
    # Check if we need to summarize old history to prevent unbounded growth
    # We summarize when history exceeds MAX_HISTORY_ROUNDS by a certain buffer (e.g., +3)
    summarization_threshold = MAX_HISTORY_ROUNDS + 3
    if len(CHAT_HISTORY) > summarization_threshold:
        # Calculate how many rounds to summarize (keep the most recent MAX_HISTORY_ROUNDS)
        rounds_to_summarize = len(CHAT_HISTORY) - MAX_HISTORY_ROUNDS
        old_rounds = CHAT_HISTORY[:rounds_to_summarize]
        
        logger.info(f"Summarizing {rounds_to_summarize} old conversation rounds...")
        new_summary = await _summarize_old_history(old_rounds, options)
        
        # Append to existing summary
        existing_summary = STATE.get("summary", "").strip()
        if existing_summary:
            STATE["summary"] = f"{existing_summary}\n\n{new_summary}"
        else:
            STATE["summary"] = new_summary
        
        # Remove summarized rounds from chat history
        del CHAT_HISTORY[:rounds_to_summarize]
        logger.info(f"Chat history trimmed from {len(CHAT_HISTORY) + rounds_to_summarize} to {len(CHAT_HISTORY)} rounds")
    
    return rendered
def _load_golden_dataset(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and isinstance(d.get("user_query"), str)]
    except Exception as e:
        logger.error(f"Failed to load dataset {path}: {e}", exc_info=True)
    return []
async def _run_golden(options: ClaudeAgentOptions, dataset_path: Path) -> None:
    entries = _load_golden_dataset(dataset_path)
    if not entries:
        logger.warning(f"Golden dataset empty/invalid: {dataset_path}")
        return
    logger.info(f"Golden dataset run start: {dataset_path} entries={len(entries)}")
    for idx, entry in enumerate(entries, 1):
        case_id = int(entry.get("id") or idx)
        user_query = str(entry["user_query"])
        logger.info(f"[golden #{case_id}] User query: {user_query}")
        print(f"[golden #{case_id}] You: {user_query}")
        rendered = await _run_turn(user_query, options)
        print(f"[golden #{case_id}] Waiter: {rendered}\n")
        logger.info(f"[golden #{case_id}] AI response: {rendered}")
        logger.info("----------------------------------------------------------")
    logger.info("Golden dataset run complete")
async def _interactive(options: ClaudeAgentOptions) -> None:
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "q":
            # Requirement: only clear state when exiting the process.
            state_reset()
            conversation_reset()
            return
        logger.info(f"User input: {user_input}")
        rendered = await _run_turn(user_input, options)
        print("\nWaiter:", rendered)
        logger.info(f"AI response: {rendered}")
        logger.info("---")
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude Agent SDK waiter demo.")
    p.add_argument("--free", action="store_true", help="Skip golden dataset autorun and start interactive mode.")
    p.add_argument("--no-interactive", action="store_true", help="Exit after golden dataset run.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (affects card validation).")
    p.add_argument("--model", type=str, default="", help="Optional Claude model override (if supported).")
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional path to golden dataset JSON (defaults to v2 if present, else legacy file).",
    )
    return p.parse_args()
async def main() -> None:
    args = _parse_args()
    _configure_logging()
    random.seed(int(args.seed))
    state_reset()
    conversation_reset()
    try:
        allowed_tools = [
            "mcp__waiter__restaurant_info_get",
            "mcp__waiter__menu_query",
            "mcp__waiter__inventory_check",
            "mcp__waiter__note_view",
            "mcp__waiter__note_clear",
            "mcp__waiter__note_load_from_order",
            "mcp__waiter__note_add_item",
            "mcp__waiter__note_set_item_quantity",
            "mcp__waiter__order_create_from_note",
            "mcp__waiter__order_update_to_match_note",
            "mcp__waiter__order_view",
            "mcp__waiter__order_process",
            "mcp__waiter__bill_calculate_total",
            "mcp__waiter__payment_charge",
        ]
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"waiter": SERVER},
            allowed_tools=allowed_tools,
            include_partial_messages=False,
            stderr=lambda s: logger.info(f"[sdk] {str(s).rstrip()}"),
        )
        if str(args.model or "").strip():
            options.model = str(args.model).strip()
        if str(args.dataset or "").strip():
            dataset_path = Path(str(args.dataset)).expanduser()
        else:
            # Force using the new dataset (the legacy dataset uses an outdated tool workflow).
            dataset_path = Path(__file__).with_name("waiter_agent_accessment_golden_dataset_v2.json")

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {dataset_path}. "
                f"Create it (recommended: waiter_agent_accessment_golden_dataset_v2.json) "
                f"or pass --dataset <path>."
            )
        if not args.free:
            await _run_golden(options, dataset_path)
            if args.no_interactive:
                return
            # Requirement: do NOT reset after dataset run; continue into interactive with last state.
        await _interactive(options)
    finally:
        # Always append usage summary at the very end of the log.
        try:
            logger.info(_TOOL_USAGE.render_report())
        except Exception:
            pass
        # Requirement: clear state before process exit.
        state_reset()
        conversation_reset()
if __name__ == "__main__":
    asyncio.run(main())
