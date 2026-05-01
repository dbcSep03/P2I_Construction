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
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
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
            "temperature": 0.6,
            "skip_special_tokens": False,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0
        }
        try:
            resp = requests.post(url, json=data, timeout=timeout)
            if resp.status_code != 200:
                print(f"[HTTP Error] Status {resp.status_code}: {resp.text[:500]}")
                try_count += 1
                continue
            j = resp.json()
            content = j['choices'][0]['text']
            last = {"results": content}
            if j['choices'][0].get('finish_reason') == 'stop':
                break
        except Exception as e:
            print(f"[Unexpected Error] {e}")
            last = {"results": "", "reasoning": f"error:{e}"}
        try_count += 1
    return last

def p_domain_and_scenario(topic: str, encoding) -> List[Dict[str, str]]:
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

def gen_domain_and_scenario(topic: str, encoding) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    out = chat_json(p_domain_and_scenario(topic, encoding), max_tokens=32000, timeout=900)
    reasoning = ""
    doc = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        doc = out['results'].split("</think>")[1].strip()
    doc = parse_json_str(doc) or {}
    domain = doc.get("domain", {}) if isinstance(doc, dict) else {}
    scenario = doc.get("scenario", {}) if isinstance(doc, dict) else {}
    if not is_slug(domain.get("slug", "")):
        domain = {"slug": "general_domain", "title": domain.get("title", "General Domain"), "one_liner": domain.get("one_liner", "")}
    if not is_slug(scenario.get("slug", "")):
        scenario = {"slug": "sample_scenario", "title": scenario.get("title", "Sample Scenario"), "overview": scenario.get("overview", "")}
    return domain, scenario, reasoning


def p_tools(topic: str, domain_slug: str, sc_slug: str, sc_title: str, sc_overview: str, k: int, encoding) -> List[Dict[str, str]]:
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
    out = chat_json(p_tools(topic, domain['slug'], scenario['slug'], scenario['title'], scenario['overview'], k, encoding), max_tokens=32000, timeout=1200)
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
For each tool call, you MUST generate a realistic `expected_tool_response` — a JSON object representing what the tool would plausibly return after a SUCCESSFUL execution.
- The response must be **structurally complete**: include all fields the tool description promises to return.
- The response must contain **concrete, domain-appropriate values** (IDs, timestamps, status codes, nested objects, arrays) — not placeholders or abstractions.
- Values in the response must be **consistent** with the tool's input arguments and the overall scenario context.
- **Subsequent steps depend on these responses:** the `produces_state_keys` and downstream `required_state_keys` must align with the actual keys and values in the expected response.
- If a tool returns a collection, include 2–3 representative items to demonstrate realistic structure.
- Include metadata fields where appropriate: timestamps, version numbers, request IDs, pagination info.

### 8. Error Response Design — failure → repair → continue (MANDATORY for 2–3 tools across the entire plan)

Select exactly 2–3 `suggested_tools` across ALL steps (not all in the same step) to carry the full **4-field error pattern**.

**Four-field structure per tool entry:**

Every tool in `suggested_tools` MUST use this structure:
- **Normal tool** (no error): 2 fields — `expected_right_tool_call` + `expected_tool_response`
- **Error tool** (failure→repair→continue): 4 fields — `expected_error_tool_call` + `expected_error_response` + `expected_right_tool_call` + `expected_tool_response`

Field semantics:
| Field | Meaning |
|---|---|
| `expected_error_tool_call` | The **initial, incorrect** tool call: `{{"name": ..., "arguments": {{...wrong params...}}}}`. Present ONLY on error tools. |
| `expected_error_response` | What the tool returns for the wrong call. Present ONLY on error tools. |
| `expected_right_tool_call` | The **corrected** tool call after the model repairs the error: `{{"name": ..., "arguments": {{...fixed params...}}}}`. Present on ALL tools. |
| `expected_tool_response` | What the tool returns for the correct call (success). Present on ALL tools. |

**Error taxonomy — choose errors from these three layers, spread across layers:**

**Layer 1 — Signature / Schema** (API intercepts the call before execution)
- `MISSING_REQUIRED_FIELD`: A required parameter is absent or a nested required subfield is missing.
- `TYPE_MISMATCH`: Parameter has wrong type (e.g., string passed where integer expected, object passed where array expected).
- `INVALID_FORMAT`: Value violates a format constraint (date not ISO 8601, UUID malformed, email missing domain, URL missing scheme).
- `CONSTRAINT_VIOLATION`: Value violates a range, length, regex, or enum constraint (e.g., page_size=5000 when max is 1000, slug contains uppercase).

**Layer 2 — World-state / Dependency** (state propagation breaks in multi-turn)
- `DANGLING_REFERENCE`: The referenced token, session, or resource ID has expired or been deleted.
- `STALE_VERSION`: A config, schema, or API version referenced is no longer active; a newer version supersedes it.
- `DEPENDENCY_NOT_SATISFIED`: A tool is called before its prerequisite tool has been called (e.g., querying before connecting).

**Layer 3 — Execution / Environment** (real system failures)
- `PERMISSION_DENIED`: Caller lacks the required role, scope, or policy clearance (HTTP 403).
- `NOT_FOUND`: The referenced entity does not exist or has been renamed/moved (HTTP 404/410).
- `RATE_LIMIT_EXCEEDED`: Call frequency or quota exceeded (HTTP 429).
- `PRECONDITION_FAILED`: Resource is in the wrong state for this operation (HTTP 409/422) — e.g., account locked, order already shipped.
- `SERVICE_UNAVAILABLE`: A downstream service is temporarily unavailable or the request timed out (HTTP 503/504).

**`expected_error_response` format — ALWAYS use this exact structure:**
{{
  "error_type": "<MUST be EXACTLY one of the 12 values listed above>",
  "http_status": <integer HTTP status code>,
  "message": "<Human-readable sentence: what failed, the bad value, and how to fix it>",
  "details": {{
    /* domain-specific diagnostic fields, e.g.:
       "received": <the bad value the caller sent>,
       "expected": <what the system expected>,
       "allowed_values": [...],
       "retry_after_seconds": 47,
       "hint": "..." */
  }}
}}

**Allowed `error_type` values (choose ONLY from this list, no other values permitted):**
"MISSING_REQUIRED_FIELD" | "TYPE_MISMATCH" | "INVALID_FORMAT" | "CONSTRAINT_VIOLATION" |
"DANGLING_REFERENCE" | "STALE_VERSION" | "DEPENDENCY_NOT_SATISFIED" |
"PERMISSION_DENIED" | "NOT_FOUND" | "RATE_LIMIT_EXCEEDED" | "PRECONDITION_FAILED" | "SERVICE_UNAVAILABLE"

**Design rules for error tools:**
- `expected_error_tool_call.arguments` must contain the WRONG value that causes the error (e.g., deprecated version, expired token, wrong type).
- `expected_right_tool_call.arguments` must contain the CORRECTED value after the model diagnoses and repairs.
- The two calls share the same `name` but differ in their `arguments`.
- Each error tool must use a DIFFERENT `error_type` — do NOT repeat the same error code across the plan.
- Place errors at realistic mid-workflow points (not always step 1) to interrupt a dependency chain.
- `expected_tool_response` must be consistent with `expected_right_tool_call.arguments`.

## Plan Specification

Design EXACTLY {rounds} ordered steps forming ONE coherent end-to-end workflow.

Each step must contain:
| Field | Description |
|---|---|
| `step_index` | Sequential number starting from 1 |
| `purpose` | Concise goal with measurable outcome |
| `suggested_tools` | Array of 1–3 tool objects; each has `expected_right_tool_call` + `expected_tool_response`; 2–3 across the plan also have `expected_error_tool_call` + `expected_error_response` |
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
          /* Normal tool — 2 fields only */
          "expected_right_tool_call": {{
            "name": "authenticate_operator",
            "arguments": {{
              "operator_id": "usr_90341A",
              "auth_mode": "certificate",
              "certificate_path": "/etc/ssl/ops/usr_90341A.pem"
            }}
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
          /* Error tool — all 4 fields: wrong call → error → right call → success */
          "expected_error_tool_call": {{
            "name": "load_environment_config",
            "arguments": {{
              "env_name": "production_east",
              "config_version": "v2.4.0"
            }}
          }},
          "expected_error_response": {{
            "error_type": "STALE_VERSION",
            "http_status": 409,
            "message": "Config version 'v2.4.0' is no longer active. The current active version is 'v2.4.1'. Please retry with the correct version.",
            "details": {{
              "received": "v2.4.0",
              "current_active_version": "v2.4.1",
              "deprecated_at": "2025-02-01T00:00:00Z",
              "hint": "Use config_version='v2.4.1'"
            }}
          }},
          "expected_right_tool_call": {{
            "name": "load_environment_config",
            "arguments": {{
              "env_name": "production_east",
              "config_version": "v2.4.1"
            }}
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
      "step_query": "I need to log in as operator usr_90341A using my certificate and load the production-east environment config.",
      "state_details": {{
        "pre_conditions": ["Operator certificate exists at /etc/ssl/ops/usr_90341A.pem", "production_east environment is available"],
        "post_conditions": ["Session token sess_a7f3d2e1 is active with read/write/execute permissions", "Environment config v2.4.1 is loaded with audit logging enabled"],
        "concrete_values": {{"session_token": "sess_a7f3d2e1", "env_id": "env_prod_east_07", "database_endpoint": "db-prod-east.internal:5432"}},
        "parameter_sources": {{"operator_id": "user-provided", "env_name": "user-provided", "config_version": "corrected after STALE_VERSION error — changed from v2.4.0 to v2.4.1"}}
      }},
      "attention_traps": ["config_version must be 'v2.4.1' not 'v2.4.0' — the old version was deprecated", "session_token is 'sess_a7f3d2e1' — do not confuse with operator_id"]
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

def gen_plan(topic: str, domain: Dict[str, Any], scenario: Dict[str, Any], tools: List[Dict[str, Any]], rounds: int, encoding=None) -> List[Dict[str, Any]]:
    out = chat_json(p_planner(topic, domain['slug'], scenario['slug'], scenario['title'], scenario['overview'], tools, rounds, encoding), max_tokens=32000, timeout=1500)
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
                state_details = st.get("state_details", {})
                if not isinstance(state_details, dict):
                    state_details = {}

                raw_tools = st.get("suggested_tools", st.get("tools", []))
                if not isinstance(raw_tools, list):
                    raw_tools = []
                suggested_tools = []
                for tool_item in raw_tools:
                    if not isinstance(tool_item, dict):
                        continue
                    parsed_tool = {}

                    # ── Error tool path: expected_error_tool_call present ──
                    if "expected_error_tool_call" in tool_item:
                        parsed_tool["expected_error_tool_call"] = tool_item["expected_error_tool_call"]

                        # Normalise error_response: accept legacy "error_code" key
                        err = tool_item.get("expected_error_response", {})
                        if isinstance(err, dict) and "error_type" not in err and "error_code" in err:
                            err["error_type"] = err.pop("error_code")
                        parsed_tool["expected_error_response"] = err

                    # ── Both paths: right tool call + success response ──
                    if "expected_right_tool_call" in tool_item:
                        parsed_tool["expected_right_tool_call"] = tool_item["expected_right_tool_call"]
                    elif "name" in tool_item:
                        # Legacy fallback: reconstruct from flat name/arguments fields
                        parsed_tool["expected_right_tool_call"] = {
                            "name": tool_item.get("name", ""),
                            "arguments": tool_item.get("arguments", {}),
                        }

                    if "expected_tool_response" in tool_item:
                        parsed_tool["expected_tool_response"] = tool_item["expected_tool_response"]

                    if parsed_tool:
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
    tdir = os.path.join(out_dir, f"{topic_slug}", f"round_{round_index}")
    os.makedirs(tdir, exist_ok=True)
    encoding = AutoTokenizer.from_pretrained("/home/dongbingcheng/Agent_FC/ckpt/Qwen3-32B")
    domain, scenario, domain_and_scenario_reasoning = gen_domain_and_scenario(topic, encoding)
    desired_rounds = max(nrounds_min, min(nrounds_max, random.randint(nrounds_min, nrounds_max)))
    k = max(tools_min, min(tools_max, random.randint(tools_min, tools_max)))

    system_info = {
        "role": "system",
        "scenario": scenario['title'],
        "domain": domain['title'] if domain.get('title') else domain['slug'],
        "topic": topic,
        "tools": [],
        "planning": []
    }
    tools = gen_tools(topic, domain, scenario, k, encoding)
    system_info["tools"] = tools
    if len(tools) != k:
        reason = f"tools_generated_mismatch: expect {k}, got {len(tools)}"
        mark_bad_case(tdir, reason)
        return {"status": "failed", "reason": reason, "stage": "tool_generation"}, [system_info]

    plan = gen_plan(topic, domain, scenario, tools, desired_rounds, encoding)
    system_info["planning"] = plan

    # Validate that at least 2 tools across the plan carry the full error pattern
    error_tool_count = sum(
        1
        for step in plan
        for tool in step.get("suggested_tools", [])
        if "expected_error_tool_call" in tool
    )
    if error_tool_count < 2:
        reason = f"insufficient_error_tools: need >=2, got {error_tool_count}"
        mark_bad_case(tdir, reason)
        return {"status": "failed", "reason": reason, "stage": "plan_generation"}, [system_info]

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
    messages_all: List[Optional[str]] = [None] * len(tasks)

    def _job(i: int, task: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        try:
            result, messages = run(task["topic"], out_dir, tools_min, tools_max, nrounds_min, nrounds_max, task["round"])
            if isinstance(result, dict) and result.get("status") == "failed":
                return i, None, messages
            else:
                return i, result, messages
        except Exception as e:
            return i, None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as ex:
        future_map = {ex.submit(_job, i, tasks[i]): i for i in range(len(tasks))}
        for fut in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map)):
            i = future_map[fut]
            try:
                idx, path, messages = fut.result()
                results[idx] = path
                messages_all[idx] = messages
            except Exception as e:
                results[i] = None

    return results, messages_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/ErrorResponse/plan_1/merged_out")
    parser.add_argument("--tools_min", type=int, default=5, help="Min tools per scenario")
    parser.add_argument("--tools_max", type=int, default=10, help="Max tools per scenario")
    parser.add_argument("--rounds_min", type=int, default=3, help="Min conversation rounds")
    parser.add_argument("--rounds_max", type=int, default=6, help="Max conversation rounds")
    parser.add_argument("--repeated", type=int, default=4000, help="Repeat each category N times")
    parser.add_argument("--max_threads", type=int, default=512, help="Max parallel threads")
    parser.add_argument("--batch_size", type=int, default=512, help="Tasks per batch")
    parser.add_argument("--batches_dir", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/ErrorResponse/plan_1/batch", help="Directory to store per-batch metadata JSONs")
    parser.add_argument("--dialogues_jsonl", type=str, default="/home/dongbingcheng/Agent_FC/plan_0208/ErrorResponse/plan_1/merged_dialogues_modified.jsonl", help="Append all dialogues into a single JSONL file")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    topics = CATEGORIES
    tasks = build_tasks(topics, args.repeated)

    os.makedirs(args.batches_dir, exist_ok=True)
    existing_batches = sorted(
        [f for f in os.listdir(args.batches_dir) if f.endswith('.json') and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )
    have_batches = len(existing_batches)
    print(f"Existing batch files detected: {have_batches}")

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

            batch_results, batch_messages_all = concurrent_run(batch_tasks, args.out_dir, args.tools_min, args.tools_max, args.rounds_min, args.rounds_max, args.max_threads)
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
            batch_messages_all = [temp for temp in batch_messages_all if temp is not None]
            batch_file = os.path.join(args.batches_dir, f"{batch_idx + 1}.json")
            with open(batch_file, 'w', encoding='utf-8') as bf:
                json.dump(batch_messages_all, bf, ensure_ascii=False, indent=4)
            print(f"Saved batch {batch_idx + 1} metadata to {batch_file}")

    print(f"Done. {completed}/{total} tasks completed. Base dir: {args.out_dir}")

if __name__ == "__main__":
    main()
