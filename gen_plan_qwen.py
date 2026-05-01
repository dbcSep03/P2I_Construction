import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
import requests
import concurrent.futures
from tqdm.auto import tqdm
import random
from prompt import USER_SIM_SYSTEM, AGENT_SYSTEM_TEMPLATE, TOOL_SIM_SYSTEM
# Built-in categories (topics)
CATEGORIES: List[str] = [
    "databases", "image-and-video-processing", "cloud-platforms", "calendar-management", "cryptocurrency", "vector-databases", "location-services", "communication", "shell-access", "multimedia-processing", "file-systems", "web-scraping", "ecommerce-and-retail", "search", "customer-data-platforms", "app-automation", "developer-tools", "os-automation", "health-and-wellness", "virtualization", "version-control", "cloud-storage", "entertainment-and-media", "games-and-gamification", "AIGC", "travel-and-transportation", "note-taking", "browser-automation", "rag-systems", "language-translation", "social-media", "security-and-iam", "home-automation-and-iot", "monitoring", "research-and-data", "art-and-culture", "customer-support", "blockchain", "finance", "knowledge-and-memory", "speech-processing", "marketing", "enterprise_business_intelligence", "transportation_logistics", "iphone_android", "smart_home", "real_estate_property", "software_apps", "legal_compliance", "education_elearning", "robot_control", "agriculture_environmental", "healthcare_medical", "manufacturing_industrial_iot", "desktop_systems", "financial_trading", "website_control", "gaming_entertainment"
]

def mark_bad_case(out_dir: str, reason: str) -> None:
    try:
        with open(os.path.join(out_dir, "bad_case.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "bad_case", "reason": reason}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def parse_json_str(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # Try to extract JSON array wrapped in square brackets []
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        # Try to extract balanced curly braces {}
        start = s.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i+1])
                        except Exception:
                            pass
        return None

def is_slug(s: str) -> bool:
    SLUG_RE = re.compile(r"^[a-z0-9_]{3,64}$")
    return bool(SLUG_RE.fullmatch((s or "").strip()))

def build_history_text(messages: List[Dict[str, Any]]) -> str:
    history_summary = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            history_summary += f"User: {content}\n"
        if role == "tool_call":
            history_summary += f"Assistant: {json.dumps(content, ensure_ascii=False)[:400]}\n"
        if role == "tool_response":
            history_summary += f"Tool: {json.dumps(content, ensure_ascii=False)[:400]}\n"
        if role == "assistant":
            history_summary += f"Assistant: {content}\n"
    return history_summary


def chat_json(messages: List[int], max_tokens: int = 8192, retries: int = 1, timeout=300) -> Any:
    try_count = 0
    last = {}
    while try_count < retries:
        
        url = "http://localhost:8000/v1/completions"
        data = {
            "model": "/home/dongbingcheng/Agent_FC/ckpt/Qwen3-32B",
            "prompt": messages,
            "max_tokens": max_tokens,
            "temperature" : 0.6,
            "skip_special_tokens": False,  
            "top_p": 0.95,
            "top_k": 20,
            "min_p":0
        }
        try:
            resp = requests.post(url, json=data, timeout=timeout)
            # resp = requests.post(url, json=data)
            
            # 检查HTTP状态码
            if resp.status_code != 200:
                print(f"[HTTP Error] Status {resp.status_code}: {resp.text[:500]}")
                try_count += 1
                continue
            
            j = resp.json()
            content = j['choices'][0]['text']
            last = {"results": content}
            # import pdb; pdb.set_trace()
            if j['choices'][0].get('finish_reason') == 'stop':
                break

        except Exception as e:
            print(f"[Unexpected Error] {e}")
            last = {"results": "", "reasoning": f"error:{e}"}
            

        try_count += 1
    return last

def p_domain_and_scenario(topic: str,encoding) -> List[Dict[str, str]]:
    instruction = (
            "You are an expert scenario designer. Return ONLY JSON.\n"
            f"Topic: \"{topic}\"\n\n"
            "Task: Propose ONE practical domain and ONE concrete scenario under this topic.\n"
            "Output JSON:\n"
            "{\n  \"topic\": \"\",\n  \"domain\": {\"slug\": \"snake_case\", \"title\": \"\", \"one_liner\": \"\"},\n  \"scenario\": {\"slug\": \"snake_case\", \"title\": \"\", \"overview\": \"\"}\n}"
        )
    convo = [
        {"role": "user", "content": instruction},
    ]
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    return tokens

def gen_domain_and_scenario(topic: str,encoding) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    # import pdb;pdb.set_trace()
    out = chat_json(p_domain_and_scenario(topic,encoding),max_tokens=32000,timeout=900)
    # import pdb;pdb.set_trace()
    # 没有调用工具，简单的问答，包含analysis和finnal两个部分
    reasoning = ""
    doc = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        doc = out['results'].split("</think>")[1].strip()
    doc = parse_json_str(doc) or {}
    # reasoning = out.get("reasoning", "")
    domain = doc.get("domain", {}) if isinstance(doc, dict) else {}
    scenario = doc.get("scenario", {}) if isinstance(doc, dict) else {}
    if not is_slug(domain.get("slug", "")):
        domain = {"slug": "general_domain", "title": domain.get("title", "General Domain"), "one_liner": domain.get("one_liner", "")}
    if not is_slug(scenario.get("slug", "")):
        scenario = {"slug": "sample_scenario", "title": scenario.get("title", "Sample Scenario"), "overview": scenario.get("overview", "")}
    return domain, scenario, reasoning


def p_tools(topic: str, domain_slug: str, sc_slug: str, sc_title: str, sc_overview: str, k: int,encoding) -> List[Dict[str, str]]:
    instruction = (
            f"""
You design production-ready function tools for complex multi-step workflows. Return ONLY JSON.
Topic: "{topic}"
Domain: "{domain_slug}"
Scenario: "{sc_title}" (slug: {sc_slug})

Scenario overview:
{sc_overview}

Task:
Produce EXACTLY {k} atomic, practical tools that form 1–2 COHESIVE, INTERDEPENDENT WORKFLOWS.

CRITICAL DESIGN PRINCIPLES
===========================

1. STATE-AWARE TOOLS
   - Tools must track and modify domain-relevant state
   - Include parameters that reference outputs from previous tool invocations
   - Design tools that REQUIRE specific outputs from other tools (create dependencies)
   - Examples across domains:
     * Database: "create_connection" returns connection_id → later queries need this ID
     * Healthcare: "authenticate_provider" returns access_token → later operations need token
     * Finance: "initiate_transaction" returns transaction_id → later tools check/confirm this ID
     * IoT: "register_device" returns device_handle → later commands use this handle
     * Manufacturing: "start_production_run" returns batch_id → tracking/quality tools need batch_id

2. FINE-GRAINED PARAMETERS
   - Use SPECIFIC, domain-appropriate parameter names
   - Include validation requirements and constraints in descriptions
   - Add parameters that capture important details and options
   - Examples across domains:
     * Database: "table_name_with_schema" (e.g., "public.users"), "isolation_level", "timeout_seconds"
     * Medical: "patient_mrn", "icd10_diagnosis_code", "medication_ndc_code", "dosage_amount_mg"
     * Finance: "account_number", "routing_number", "transaction_type", "currency_iso_code"
     * Robotics: "joint_position_degrees", "velocity_limit_mps", "coordinate_frame", "safety_zone_id"

3. TOOL INTERDEPENDENCIES
   - Design tool chains where outputs of one tool are required inputs for subsequent tools
   - Include "returns" specification in descriptions: exact keys and value types returned
   - Create complementary tool types:
     * Query/getter tools: retrieve current state, list resources, check status
     * Mutation/setter tools: create, update, delete, modify state
     * Validation tools: verify prerequisites, check constraints, validate data
     * Execution tools: perform operations using state from query/validation tools
   - Ensure tools span the operation lifecycle: preparation → validation → execution → verification

4. COMPLEXITY LEVELS (MIX)
   - Basic tools: atomic operations (check, get, retrieve single items)
   - Medium tools: operations requiring context or previous results (create with config, update based on query)
   - Advanced tools: operations with complex dependencies, side effects, or state transitions (batch operations, transactions, workflows)
   - Design tools that naturally encourage multi-tool usage

Coherence requirements:
- All tools belong to ONE problem domain with natural composition patterns
- Tools form clear workflows: preparation → execution → validation → cleanup
- Include complementary tool pairs: 
  * Query tools (get_status, list_items) + Action tools (update, delete)
  * Validation tools (check_prerequisites, verify_integrity) + Execution tools (run, process)
- Cover edge cases: error recovery, rollback, status checking

Rules:
- name: snake_case, 3–64 chars, descriptive of exact function
- description: 
  * First sentence: what it does
  * Second sentence: what it returns (be specific: "Returns {{'task_id': str, 'status': str}}")
  * Third sentence (optional): preconditions or dependencies
- parameters:
  - "type": "dict"
  - "properties": realistic, detailed fields with:
    * "type": string|number|integer|boolean|array|dict
    * "description": include expected format, constraints, and source (e.g., "from create_session result")
    * "default": provide when sensible
  - "required": non-empty list of essential fields (at least 1-2 per tool)
- Design at least 3 tools with parameters that explicitly reference outputs from other tools

Output JSON (ONLY):
{{
  "scenario_slug": "{sc_slug}",
  "tools": [
    {{
      "function": {{
        "type": "function",
        "function": {{
          "name": "",
          "description": "",
          "parameters": {{
            "type": "dict",
            "properties": {{
              /* Use detailed, specific parameter names with full descriptions */
              /* Example: "session_id": {{"type":"string", "description":"Session ID returned by initialize_session tool"}},
                          "target_directory_path": {{"type":"string", "description":"Absolute path to target directory, must exist"}},
                          "operation_mode": {{"type":"string", "description":"Either 'move' or 'copy', affects whether source is deleted"}} */
            }},
            "required": ["param1", "param2"]
          }}
        }}
      }}
    }}
  ]
}}
"""
        )

    convo = [
        {"role": "user", "content": instruction},
    ]
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    return tokens

def gen_tools(topic: str, domain: Dict[str, Any], scenario: Dict[str, Any], k: int = 8, encoding=None) -> List[Dict[str, Any]]:
    # import pdb;pdb.set_trace()
    out = chat_json(p_tools(topic, domain['slug'], scenario['slug'], scenario['title'], scenario['overview'], k,encoding),max_tokens=32000,timeout=1200)
    # import pdb; pdb.set_trace()
    reasoning = ""
    doc = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        doc = out['results'].split("</think>")[1].strip()
    doc = parse_json_str(doc) or {}
    tools = []
    for item in (doc.get("tools", []) if isinstance(doc, dict) else []):
        try:
            fn = item["function"]["function"]
            name = fn.get("name", "").replace(' ', '_').lower()
            if not is_slug(name):
                continue
            tools.append({
                "function": {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}) or {}
                    }
                }
            })
        except Exception:
            continue
    # dedupe by name
    seen = set()
    unique = []
    for t in tools:
        nm = t["function"]["function"]["name"]
        if nm in seen:
            continue
        seen.add(nm)
        unique.append(t)
    return unique


def p_planner(topic: str, domain_slug: str, sc_slug: str, sc_title: str, sc_overview: str, tools: List[Dict[str, Any]], rounds: int, encoding=None) -> List[Dict[str, str]]:
    tool_list = []
    for t in tools:
        fn = t.get("function", {}).get("function", {})
        tool_list.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {})
        })
    catalog = json.dumps(tool_list, ensure_ascii=False)
    instruction = (
f"""You are a senior workflow architect designing rigorous, multi-step task plans. Return ONLY valid JSON.

## Context
- Topic: "{topic}"
- Domain: "{domain_slug}"
- Scenario: "{sc_title}" (slug: {sc_slug})
- Scenario Overview: {sc_overview}

## Available Tools (use ONLY these; do NOT invent new tools)
{catalog}

## Core Planning Principles

### 1. Precise State Tracking
Every step must explicitly manage state with CONCRETE values — not abstract labels.
- Use realistic identifiers: "session_id": "sess_a7f3d2e1", "account_number": "ACCT-8827431", "balance_usd": 1250.75
- Track state transitions: when a resource is created, renamed, moved, or deleted, ALL subsequent steps must reference the NEW state.
- Each step documents: what state is REQUIRED before execution, what state is PRODUCED after execution.

### 2. Parallel Multi-Tool Steps
Maximize step complexity by combining 2–3 tools per step wherever possible.
- **HARD RULE:** Tools within the SAME step must be fully independent — they may only consume outputs from PREVIOUS steps, never from each other.
- If tool B requires tool A's output, place them in SEPARATE consecutive steps.
- Good parallel patterns: independent queries, complementary validations, concurrent reads.

### 3. Domain-Realistic Details
- Use domain-authentic identifiers: `usr_90341A`, `INV-2025-0047`, `DB_analytics_v2`, `ICD10:J45.20` — not generic placeholders.
- Include precise data formats, units, constraints, and validation rules.
- All parameter values must be technically plausible and mutually consistent within the scenario.

### 4. Explicit Parameter Provenance
For every tool argument, document exactly where the value comes from:
- From user input: note the source explicitly
- From a prior step: specify step number and exact result key path (e.g., "from step 2 → create_order.response.order_id")
- Derived/computed: explain the transformation logic

### 5. Rich Dependency Graphs
- Chain dependencies: step N may depend on results from multiple earlier steps.
- Cross-references: parameters reference specific values from steps X, Y, Z.
- Design workflows where an incorrect value from an earlier step would cause a realistic downstream failure.

### 6. Attention Traps
Embed subtle precision challenges throughout the plan:
- Resource relocation: file moved → later steps must use the NEW path
- Renaming: entity renamed → all references must use the NEW name
- Case-sensitive values, similar-looking parameter names, unit conversions
- A single mistake in any identifier, field name, or format should break the workflow.

### 7. Realistic Tool Responses (CRITICAL)
For each tool call, you MUST generate a realistic `expected_tool_response` — a JSON object representing what the tool would plausibly return in a real production environment.
- The response must be **structurally complete**: include all fields the tool description promises to return.
- The response must contain **concrete, domain-appropriate values** (IDs, timestamps, status codes, nested objects, arrays) — not placeholders or abstractions.
- Values in the response must be **consistent** with the tool's input arguments and the overall scenario context.
- **Subsequent steps depend on these responses:** the `produces_state_keys` and downstream `required_state_keys` must align with the actual keys and values in the expected response.
- If a tool returns a collection, include 2–3 representative items to demonstrate realistic structure.
- Include metadata fields where appropriate: timestamps, version numbers, request IDs, pagination info.

## Plan Specification

Design EXACTLY {rounds} ordered steps forming ONE coherent end-to-end workflow.

Each step must contain:
| Field | Description |
|---|---|
| `step_index` | Sequential number starting from 1 |
| `purpose` | Concise goal with measurable outcome |
| `suggested_tools` | Array of 1–3 tool calls, each with `name`, `arguments`, and `expected_tool_response` |
| `required_state_keys` | State variables this step consumes (empty for step 1 if no prior state) |
| `produces_state_keys` | NEW state variables this step creates |
| `step_query` | Natural-language user request that would trigger this step |
| `state_details` | Object with `pre_conditions`, `post_conditions`, `concrete_values`, `parameter_sources` |
| `attention_traps` | List of common mistakes or pitfalls specific to this step |

## Tool Usage Rules
- Across steps: minimize tool repetition unless the workflow logically requires it.
- Within a step: each tool appears at most once; all tools are independent of each other.
- Every tool argument must use a concrete value from `required_state_keys`, `concrete_values`, or the user query — **no abbreviations or ellipses**.
- Every `expected_tool_response` must return concrete values that downstream steps can consume.

## Dataflow Discipline
- `required_state_keys` must reference keys from earlier `produces_state_keys`.
- First step may have empty `required_state_keys`.
- Every produced key should be consumed by at least one later step (no dangling keys).
- The values in `expected_tool_response` must be the source of truth for `produces_state_keys`.

## Output JSON
{{
  "plan": [
    {{
      "step_index": 1,
      "purpose": "Authenticate and initialize session for operator usr_90341A",
      "suggested_tools": [
        {{
          "name": "authenticate_operator",
          "arguments": {{
            "operator_id": "usr_90341A",
            "auth_mode": "certificate",
            "certificate_path": "/etc/ssl/ops/usr_90341A.pem"
          }},
          "expected_tool_response": {{
            "status": "authenticated",
            "session_token": "sess_a7f3d2e1",
            "operator_id": "usr_90341A",
            "permissions": ["read", "write", "execute"],
            "expires_at": "2025-02-09T03:22:00Z"
          }}
        }},
        {{
          "name": "load_environment_config",
          "arguments": {{
            "env_name": "production_east",
            "config_version": "v2.4.1"
          }},
          "expected_tool_response": {{
            "env_id": "env_prod_east_07",
            "region": "us-east-1",
            "database_endpoint": "db-prod-east.internal:5432",
            "config_version": "v2.4.1",
            "feature_flags": {{"enable_audit_log": true, "max_batch_size": 500}}
          }}
        }}
      ],
      "required_state_keys": [],
      "produces_state_keys": ["session_token", "operator_permissions", "env_id", "database_endpoint", "feature_flags"],
      "step_query": "I need to log in as operator usr_90341A using my certificate and load the production-east environment config v2.4.1.",
      "state_details": {{
        "pre_conditions": ["Operator certificate exists at /etc/ssl/ops/usr_90341A.pem", "production_east environment is available"],
        "post_conditions": ["Session token sess_a7f3d2e1 is active with read/write/execute permissions", "Environment config v2.4.1 is loaded with audit logging enabled"],
        "concrete_values": {{"session_token": "sess_a7f3d2e1", "env_id": "env_prod_east_07", "database_endpoint": "db-prod-east.internal:5432"}},
        "parameter_sources": {{"operator_id": "user-provided", "env_name": "user-provided", "config_version": "user-provided"}}
      }},
      "attention_traps": ["session_token is 'sess_a7f3d2e1' — do not confuse with operator_id", "config_version is 'v2.4.1' with lowercase 'v'", "database_endpoint includes port number :5432"]
    }}
  ]
}}
"""
        )
    convo = [
        {"role": "user", "content": instruction},
    ]
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    return tokens

def gen_plan(topic: str, domain: Dict[str, Any], scenario: Dict[str, Any], tools: List[Dict[str, Any]], rounds: int, encoding=None,) -> List[Dict[str, Any]]:
    # import pdb;pdb.set_trace()
    out = chat_json(p_planner(topic, domain['slug'], scenario['slug'], scenario['title'], scenario['overview'], tools, rounds, encoding),max_tokens=32000,timeout=1200)
    # import pdb; pdb.set_trace()
    reasoning = ""
    doc = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        doc = out['results'].split("</think>")[1].strip()
    doc = parse_json_str(doc) or {}
    plan = []
    if isinstance(doc, dict):
        raw = doc.get("plan", [])
        if isinstance(raw, list):
            for i, st in enumerate(raw):
                if not isinstance(st, dict):
                    continue
                # Extract state_details with defaults
                state_details = st.get("state_details", {})
                if not isinstance(state_details, dict):
                    state_details = {}

                # Extract suggested_tools and ensure expected_tool_response is preserved
                raw_tools = st.get("suggested_tools", st.get("tools", []))
                if not isinstance(raw_tools, list):
                    raw_tools = []
                suggested_tools = []
                for tool_item in raw_tools:
                    if not isinstance(tool_item, dict):
                        continue
                    parsed_tool = {
                        "name": tool_item.get("name", ""),
                        "arguments": tool_item.get("arguments", {}),
                    }
                    # Preserve expected_tool_response if present
                    if "expected_tool_response" in tool_item:
                        parsed_tool["expected_tool_response"] = tool_item["expected_tool_response"]
                    suggested_tools.append(parsed_tool)

                plan.append({
                    "step_index": st.get("step_index", i + 1),
                    "purpose": st.get("purpose", st.get("goal", "")),
                    "suggested_tools": suggested_tools,
                    "required_state_keys": st.get("required_state_keys", []),
                    "produces_state_keys": st.get("produces_state_keys", []),
                    "step_query": st.get("step_query", ""),
                    "state_details": {
                        "pre_conditions": state_details.get("pre_conditions", []),
                        "post_conditions": state_details.get("post_conditions", []),
                        "concrete_values": state_details.get("concrete_values", {}),
                        "parameter_sources": state_details.get("parameter_sources", {})
                    },
                    "attention_traps": st.get("attention_traps", [])
                })
    return plan


def run(topic: str, out_dir: str, tools_min: int, tools_max: int, nrounds_min: int, nrounds_max: int, round_index: int) -> Optional[str]:
    topic_slug = topic.replace("/", "_")
    # # import pdb;pdb.set_trace()
    tdir = os.path.join(out_dir, f"{topic_slug}", f"round_{round_index}")
    os.makedirs(tdir, exist_ok=True)
    encoding = AutoTokenizer.from_pretrained("/home/dongbingcheng/Agent_FC/ckpt/Qwen3-32B")
    domain, scenario, domain_and_scenario_reasoning = gen_domain_and_scenario(topic,encoding) # 生成实际的应用场景
    desired_rounds = max(nrounds_min, min(nrounds_max, random.randint(nrounds_min, nrounds_max)))
    k = max(tools_min, min(tools_max, random.randint(tools_min, tools_max)))

    # 先创建system_info，即使早期失败也能返回
    system_info = {
        "role": "system",
        "scenario": scenario['title'],
        "domain": domain['title'] if domain.get('title') else domain['slug'],
        "topic": topic,
        "tools": [],
        "planning": []
    }
    # 生成工具
    tools = gen_tools(topic, domain, scenario, k, encoding) # 生成工具
    system_info["tools"] = tools
    if len(tools) != k:
        reason = f"tools_generated_mismatch: expect {k}, got {len(tools)}"
        mark_bad_case(tdir, reason)
        return {"status": "failed", "reason": reason, "stage": "tool_generation"}, [system_info]

    # 生成计划
    plan = gen_plan(topic, domain, scenario, tools, desired_rounds,encoding)
    # import pdb; pdb.set_trace()
    system_info["planning"] = plan
    if not plan or len(plan) != desired_rounds:
        reason = f"plan_steps_mismatch: expect {desired_rounds}, got {len(plan) if plan else 0}"
        mark_bad_case(tdir, reason)
        return {"status": "failed", "reason": reason, "stage": "plan_generation"}, [system_info]

    return {"status": "success", "reason": "plan_generation", "stage": "plan_generation"}, [system_info]


def build_tasks(categories: List[str], repeated: int) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for r in range(repeated):
        for idx, cat in enumerate(categories):
            tasks.append({"topic": cat, "round": r, "index": r * len(categories) + idx})
    return tasks

def concurrent_run(tasks: List[Dict[str, Any]], out_dir: str, tools_min: int, tools_max: int, nrounds_min: int, nrounds_max: int, max_threads: int) -> List[Optional[str]]:
    results: List[Optional[str]] = [None] * len(tasks)
    messages_all : List[Optional[str]] = [None] * len(tasks)
    def _job(i: int, task: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        try:
            result, messages = run(task["topic"], out_dir, tools_min, tools_max, nrounds_min, nrounds_max, task["round"]) 
            # result can be a path (str) or failure dict
            if isinstance(result, dict) and result.get("status") == "failed":
                return i, None, messages  # Failed case
            else:
                return i, result, messages  # Success case (path)
        except Exception as e:
            # print(f"[ERROR] task#{i} topic='{task.get('topic')}' failed: {e}")
            return i, None, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as ex:
        future_map = {ex.submit(_job, i, tasks[i]): i for i in range(len(tasks))}
        for fut in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map)):
            i = future_map[fut]
            try:
                idx, path,messages = fut.result()
                results[idx] = path
                messages_all[idx] = messages
            except Exception as e:
                # print(f"[ERROR] collecting result for task#{i}: {e}")
                results[i] = None
    return results, messages_all

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/Base/plan_6/merged_out")
    parser.add_argument("--tools_min", type=int, default=5, help="Min tools per scenario")
    parser.add_argument("--tools_max", type=int, default=10, help="Max tools per scenario")
    parser.add_argument("--rounds_min", type=int, default=3, help="Min conversation rounds")
    parser.add_argument("--rounds_max", type=int, default=8, help="Max conversation rounds")
    parser.add_argument("--repeated", type=int, default=4000, help="Repeat each category N times")
    parser.add_argument("--max_threads", type=int, default=512, help="Max parallel threads")
    parser.add_argument("--batch_size", type=int, default=512, help="Tasks per batch")
    parser.add_argument("--batches_dir", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/Base/plan_6/batch", help="Directory to store per-batch metadata JSONs")
    parser.add_argument("--dialogues_jsonl", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/Base/plan_6/merged_dialogues_modified.jsonl", help="Append all dialogues into a single JSONL file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    topics = CATEGORIES 
    tasks = build_tasks(topics, args.repeated)

    # batching with resume
    os.makedirs(args.batches_dir, exist_ok=True)
    existing_batches = sorted(
        [f for f in os.listdir(args.batches_dir) if f.endswith('.json') and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )
    have_batches = len(existing_batches)
    print(f"Existing batch files detected: {have_batches}")

    # open dialogues jsonl for append
    os.makedirs(os.path.dirname(os.path.abspath(args.dialogues_jsonl)) or '.', exist_ok=True)
    total = len(tasks)
    completed = 0
    with open(args.dialogues_jsonl, 'a', encoding='utf-8') as jsonl_out:
        for i in range(0, total, args.batch_size):
            batch_idx = i // args.batch_size
            if batch_idx < have_batches:
                print(f"Skipping already processed batch {batch_idx + 1}")
                continue
            batch_tasks = tasks[i:i + args.batch_size]
            print(f"Processing batch {batch_idx + 1} (tasks {i} ~ {min(i + args.batch_size, total) - 1}) ...")

            # run concurrently this batch
            batch_results, batch_messages_all = concurrent_run(batch_tasks, args.out_dir, args.tools_min, args.tools_max, args.rounds_min, args.rounds_max, args.max_threads)
            # collect successes
            success_entries: List[Dict[str, Any]] = []
            for j, path in enumerate(batch_results):
                if not path:
                    continue
                try:
                    dlg_path = os.path.join(path, 'dialogue.json')
                    with open(dlg_path, 'r', encoding='utf-8') as f:
                        conv = json.load(f)
                    jsonl_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    success_entries.append({
                        "task_index": i + j,
                        "topic": batch_tasks[j]["topic"],
                        "round": batch_tasks[j]["round"],
                        "output_dir": path
                    })
                    completed += 1
                except Exception as e:
                    print(f"[WARN] failed to append dialogue for task {i + j}: {e}")
                    continue
            jsonl_out.flush()
            # import pdb;pdb.set_trace()
            batch_messages_all = [temp for temp in batch_messages_all if temp is not None]
            # write batch metadata json
            batch_file = os.path.join(args.batches_dir, f"{batch_idx + 1}.json")
            with open(batch_file, 'w', encoding='utf-8') as bf:
                json.dump(batch_messages_all, bf, ensure_ascii=False, indent=4)
            print(f"Saved batch {batch_idx + 1} metadata to {batch_file}")

    print(f"Done. {completed}/{total} tasks completed. Base dir: {args.out_dir}")

if __name__ == "__main__":
    main()



