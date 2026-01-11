
import json
import re
import sys
from typing import Dict, List, Any, Optional

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cases = {}
    current_case_id = None
    buffer = []
    
    # Metadata extraction
    model_name = "unknown"
    run_date = "unknown"
    
    # regex patterns
    golden_start_pat = re.compile(r"INFO - \[golden #(\d+)\] User query: (.*)")
    # We might find the model name at the start
    model_pat = re.compile(r"Using NVIDIA OpenAI-compatible model: (.*)")
    date_pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    
    # Tool call pattern (from log lines like: "tool_call: get_food_menu args={}")
    tool_call_pat = re.compile(r"tool_call: (\w+) args=(.*)")
    
    # AI response pattern: "[golden #1] AI response: ..." 
    # OR standard LLM response logging if golden tag is missing in some lines, but the log seems to have specific golden tags for responses too.
    ai_response_pat = re.compile(r"INFO - \[golden #(\d+)\] AI response: (.*)")
    
    # State pattern
    state_pat = re.compile(r"Current active order \(structured, authoritative\):")
    
    # Summary pattern
    summary_pat = re.compile(r"Conversation summary so far:")

    # Regex for detecting block starts
    llm_response_start = re.compile(r"INFO - LLM response:")
    log_line_start = re.compile(r"^\d{4}-\d{2}-\d{2}")

    # State for parsing
    in_response_block = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # 1. Extract Run Info
        if run_date == "unknown":
            m_date = date_pat.match(line)
            if m_date:
                run_date = m_date.group(1)
        
        m_model = model_pat.search(line)
        if m_model:
            model_name = m_model.group(1).strip()

        # Check for new log line to reset block state (unless it is the start of a response block itself)
        if log_line_start.match(line):
            if llm_response_start.search(line):
                in_response_block = True
            else:
                in_response_block = False
            
        # 2. Detect Case Start
        m_start = golden_start_pat.search(line)
        if m_start:
            case_id = int(m_start.group(1))
            current_case_id = case_id
            cases[case_id] = {
                "user_query": m_start.group(2),
                "tool_calls": [],
                "ai_response": "",
                "logs": [],
                "state_snapshots": [],
                "summaries": []
            }
            # Start of case also resets response block
            in_response_block = False
            continue
            
        if current_case_id is not None:
            # Check if we moved to next case or finished
            if "Golden dataset run complete" in line:
                current_case_id = None
                in_response_block = False
                continue
            
            # Capture content for current case
            cases[current_case_id]["logs"].append(line_stripped)
            
            # Extract Tool Calls ONLY if in response block
            if in_response_block:
                m_tool = tool_call_pat.search(line)
                if m_tool:
                    cases[current_case_id]["tool_calls"].append({
                        "name": m_tool.group(1),
                        "args": m_tool.group(2)
                    })
            
            # Extract AI Response (This specific golden log line appears regardless of block state, typically)
            m_ai = ai_response_pat.search(line)
            if m_ai:
                 if int(m_ai.group(1)) == current_case_id:
                     cases[current_case_id]["ai_response"] = m_ai.group(2)
            
            # Extract State
            if state_pat.search(line):
                if i + 1 < len(lines):
                    try:
                        state_json = json.loads(lines[i+1].strip())
                        cases[current_case_id]["state_snapshots"].append(state_json)
                    except:
                        pass

            # Extract Summary
            if summary_pat.search(line):
                cases[current_case_id]["summaries"].append("summary_detected")

    return {
        "meta": {
            "model_name": model_name,
            "run_date": run_date,
            "log_path": log_path
        },
        "cases": cases
    }

def evaluate_cases(parsed_data, golden_data):
    results = []
    golden_map = {item['id']: item for item in golden_data}
    
    for case_id, case_data in parsed_data["cases"].items():
        golden = golden_map.get(case_id)
        if not golden:
            continue
            
        expected = golden.get("expected_behavior", "")
        key_info = golden.get("key_information_needed", [])
        
        # 1. Check Tool Usage
        executed_tools = [t['name'] for t in case_data['tool_calls']]
        
        # Simple keyword matching for tools in expected behavior
        # e.g. "Call 'get_food_menu'" -> check if get_food_menu in executed_tools
        tool_status = "OK"
        missing_tools = []
        
        for k in key_info:
            if "Tool Call:" in k:
                required_tool = k.split(":")[1].split("(")[0].strip()
                # Remove quotes if present
                required_tool = required_tool.replace("'", "").replace('"', "")
                
                # Check match (lenient)
                match = False
                for ft in executed_tools:
                    if required_tool in ft:
                        match = True
                        break
                if not match:
                    missing_tools.append(required_tool)
        
        if missing_tools:
            tool_status = f"MISSING: {', '.join(missing_tools)}"
        
        # 2. Check AI Response Keywords
        response_text = case_data["ai_response"]
        missing_keywords = []
        # Filter key_info to ignore Tool Calls for this check
        text_keys = [k for k in key_info if "Tool Call" not in k and "Normalization" not in k]
        
        # Very basic check
        # for k in text_keys:
        #    if k.lower() not in response_text.lower():
        #        missing_keywords.append(k)
        
        # 3. Determine Pass/Fail
        # Pass = No missing tools. (Text exact match is too hard, we rely on semantic review in report)
        status = "✅"
        comment = "Pass"
        
        if missing_tools:
            status = "❌"
            comment = f"Missing tools: {missing_tools}"
        elif len(executed_tools) > 3 and case_id not in [19, 20]: # Heuristic for inefficiency
             status = "⚠️"
             comment = "Potential inefficiency (many tools)"
             
        # Case specific checks
        if case_id == 13: # Inventory check
             if "insufficient_stock" not in str(case_data['logs']) and "5" not in response_text:
                 status = "❌"
                 comment = "Failed to detect insufficient stock"
        
        results.append({
            "id": case_id,
            "query": case_data["user_query"],
            "status": status,
            "comment": comment,
            "tools": executed_tools,
            "response_snippet": response_text[:100] + "..." if len(response_text) > 100 else response_text
        })
        
    return results

if __name__ == "__main__":
    log_file = "waiter_react_agent.log"
    golden_file = "waiter_agent_accessment_golden_dataset.json"
    
    data = parse_log(log_file)
    golden = load_json(golden_file)
    
    eval_results = evaluate_cases(data, golden)
    
    # Output JSON for the agent to read
    output_path = "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": data["meta"],
            "results": eval_results,
            "raw_cases": data["cases"]
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Output written to {output_path}")
