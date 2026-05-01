import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional, Tuple
import requests
from transformers import AutoTokenizer
import concurrent.futures
from tqdm.auto import tqdm
import random
from prompt import USER_SIM_SYSTEM, AGENT_SYSTEM_TEMPLATE, TOOL_SIM_SYSTEM,qwen_tool_response_template, qwen_tool_call_template

def gen_syten_message_qwen(tools):
    system_prompt = """\
A conversation between User and Assistant. The User asks a question, and the Assistant solves it.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{Tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Thinking and Tool-Calling Protocol

You must follow this structured thinking process strictly.

## Before calling tool(s):
Think about whether calling a tool is necessary for the user's current request, and if so, which tool to call and what arguments to pass. Keep it brief (2-5 sentences).

## After receiving tool response(s):
Think about whether the user's current request has been fulfilled. If fulfilled, end thinking and provide a concise summary. If not, continue with the next necessary tool call.

## Critical Rules:
- SCOPE: Only call tools that the user's CURRENT message explicitly requires. Never proactively call tools for anticipated "next steps" or future needs the user has not asked for.
- BREVITY: Keep each thinking block to 2-5 sentences. Do not speculate, deliberate at length, or narrate your reasoning process.
- STOP CONDITION: Once the user's request is fulfilled, immediately stop and summarize. Do not continue with additional unrequested tool calls.
- PRECISION: Use exact identifiers, IDs, and values from prior tool responses — do not abbreviate or guess."""
    tools_info = "\n".join([json.dumps(temp) for temp in tools])
    return system_prompt.replace("{Tools}", tools_info)



def mark_bad_case(out_dir: str, reason: str) -> None:
    try:
        with open(os.path.join(out_dir, "bad_case.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "bad_case", "reason": reason}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def extract_tool_calls(content,start_tag="<tool_call>", end_tag="</tool_call>"):
    """
    Extract tool calls from the content.
    """
    if '</think>' in content:
        content = content.split("</think>")[1].strip()
    tool_calls = []
    start_index = content.find(start_tag)
    while start_index != -1:
        end_index = content.find(end_tag, start_index)
        if end_index == -1:
            break
        tool_call = content[start_index + len(start_tag):end_index]
        try:
            tool_calls.append(json.loads(tool_call))
        except json.JSONDecodeError:
            # print(f"Error decoding JSON from tool call: {tool_call}")
            pass
        # Move to the next tool call
        start_index = content.find(start_tag, end_index + len(end_tag))
    
    return tool_calls

def convert_old_to_new(old_func: dict) -> dict:
    func = old_func.get("function", old_func)
    params = func.get("parameters", {})
    if params.get("type") == "dict":
        params["type"] = "object"
    return {
        "name": func["name"],
        "description": func.get("description", ""),
        "parameters": params,
    }

def render_tool_catalog(tools: List[Dict[str, Any]]) -> str:
    return "\n".join([json.dumps(temp) for temp in tools])

def validate_tool_sim_json(doc: Any, n_calls: int) -> Tuple[bool, str]:
    # Check if it's the new format (list of tool results)
    if isinstance(doc, list):
        if len(doc) != n_calls:
            return False, f"Tool results length {len(doc)} must match # of tool calls {n_calls}."
        for i, r in enumerate(doc):
            if not isinstance(r, dict):
                return False, f"Tool result[{i}] not object."
            if "name" not in r or "results" not in r:
                return False, f"Tool result[{i}] missing 'name' or 'results' fields."
        return True, ""
    
    # Check if it's the old format (for backward compatibility)
    if not isinstance(doc, dict):
        return False, "Tool sim output not a JSON object or list."
    
    if "results" not in doc:
        return False, "Missing 'results'."
    tr = doc["results"]
    if not isinstance(tr, list) or len(tr) != n_calls:
        return False, "results length must match # of tool calls."
    for i, r in enumerate(tr):
        if not isinstance(r, dict):
            return False, f"results[{i}] not object."
        if "name" not in r or "results" not in r:
            return False, f"results[{i}] missing 'name' or 'results' fields."
    return True, ""

def parse_json_str(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # 尝试提取中括号 [] 包裹的 JSON 数组
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        # 尝试提取平衡的大括号 {}
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

def create_history_summary(conversation_history: List[Dict[str, Any]], hint_prompt: str) -> str:
    return_conv = []
    for t in conversation_history:
        if t["role"] == "user":
            return_conv.append({"role": "user", "content": t['content']})
        if t["role"] == "tool_call":
            return_conv.append({"role": "assistant", "content": '<think>\n\n<think>' + '\n'.join([f'<tool_call>\n{json.dumps(tool, ensure_ascii=False)}\n</tool_call>' for tool in t['content']]) })
        if t["role"] == "tool_response":
            return_conv.append({"role": "user", "content": '\n'.join([f'<tool_response>\n{json.dumps(tool, ensure_ascii=False)}\n</tool_response>' for tool in t['content']])})
        if t['role'] == "assistant":
            return_conv.append({"role": "assistant", "content": '<think>\n\n<think>' + t['content']})
    if len(hint_prompt) > 0:
        if return_conv and return_conv[-1]['role'] == "user":
            return_conv[-1]['content'] += hint_prompt
        else:
            return_conv.append({"role": "user", "content": hint_prompt})
    return return_conv

def chat_json(messages: List[int], max_tokens: int = 8192, retries: int = 2, timeout=300) -> Any:
    try_count = 0
    last = {}
    while try_count < retries:
        
        url = "http://localhost:8000/v1/completions"
        data = {
            # "model": "/home/dongbingcheng/Agent_FC/gpt-oss-120b",
            "prompt": messages,
            "max_tokens": max_tokens,
            "temperature" : 0.6,
            "skip_special_tokens": False,  
        }
        try:
            # resp = requests.post(url, json=data, timeout=timeout)
            resp = requests.post(url, json=data)
            
            # 检查HTTP状态码
            if resp.status_code != 200:
                print(f"[HTTP Error] Status {resp.status_code}: {resp.text[:500]}")
                try_count += 1
                continue
            
            j = resp.json()
            content = j['choices'][0]['text']
            # import pdb; pdb.set_trace()
            # sglang是finish_reason 而vllm是stop_reason
            # if j['choices'][0]['finish_reason'] == 200012:
            #     content += "<|call|>"
            last = {"results": content}
            # import pdb; pdb.set_trace()
            if j['choices'][0].get('finish_reason') == 'stop':
                break

        except Exception as e:
            print(f"[Unexpected Error] {e}")
            last = {"results": "", "reasoning": f"error:{e}"}
            

        try_count += 1
    return last

def run_user_query_for_step(user_system: str,
                           tool_catalog: str,
                           scenario: str,
                           domain: str,
                           topic: str,
                           history_hint: str,
                           conversation_history: List[Dict[str, Any]],encoding) -> Tuple[str, str]:
    # import pdb;pdb.set_trace()
    history_summary = build_history_text(conversation_history)

    tool_info = ""
    if tool_catalog:
        tool_info = f"\nAvailable tools that the assistant can use:\n{tool_catalog}\nYou can ask questions that require the assistant to use these tools.\n"

    instruction = (
            f"Current conversation context:\n- Scenario: {scenario}\n- Domain: {domain}\n- Topic: {topic}\n"
            f"- Current status hint: {history_hint}\n"
            f"{history_summary}\n{tool_info}"
            "Based on the above information, please directly output your next question or request (no JSON format needed, just output natural language)."
            "Please use the query from **this step**, and do not go beyond the **purpose of the current round**."
        )
    convo = [
        {"role": "system", "content": user_system},
        {"role": "user", "content": instruction}
    ]
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    out = chat_json(tokens, max_tokens=16000,timeout=900)
    # import pdb;pdb.set_trace()
    reasoning = ""
    result = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        result = out['results'].split("</think>")[1].strip()
    if result and not result.startswith("{") and not result.startswith("["):
        return result, reasoning
    # fallback
    return ("Please continue with the next step based on prior results.", reasoning)

def build_tool_responses_with_expected(
    tool_calls: List[Dict[str, Any]],
    expected_responses_map: Dict[str, Any],
    tools: List[Dict[str, Any]],
    world_state: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    st: Dict[str, Any],
    encoding
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    为 agent 的 tool calls 构建响应：
    1. 如果 tool call 名称匹配 expected_tool_response，直接使用预期的返回值
    2. 如果不匹配，退回到模型模拟生成
    返回: (tool_responses 列表, reasoning 字符串)
    """
    results = [None] * len(tool_calls)
    unmatched_indices = []
    unmatched_calls = []
    matched_count = 0

    # 第一轮：用 expected_tool_response 匹配
    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        if name in expected_responses_map:
            results[i] = {"name": name, "results": expected_responses_map[name]}
            matched_count += 1
        else:
            unmatched_indices.append(i)
            unmatched_calls.append(tc)

    # 第二轮：对未匹配的 tool calls 使用模型模拟
    sim_reasoning = ""
    if unmatched_calls:
        sim_results, sim_reasoning = simulate_single_tool_call(
            unmatched_calls, tools, world_state, conversation_history, st, encoding
        )
        if sim_results is None:
            # 模拟失败，但已匹配的部分仍可用；如果全部未匹配才返回 None
            if matched_count == 0:
                return None, ""
            # 对失败的模拟用空结果填充
            for idx in unmatched_indices:
                if results[idx] is None:
                    tc_name = tool_calls[idx].get("name", "unknown")
                    results[idx] = {"name": tc_name, "results": {"error": True, "message": "Tool simulation failed"}}
        else:
            for j, idx in enumerate(unmatched_indices):
                if j < len(sim_results):
                    results[idx] = sim_results[j]

    # 最终检查：确保所有位置都有结果
    if any(r is None for r in results):
        return None, ""

    reasoning = f"matched={matched_count}/{len(tool_calls)}"
    if sim_reasoning:
        reasoning += f" | sim: {sim_reasoning}"
    return results, reasoning


def simulate_single_tool_call(tool_calls: List[Dict[str, Any]], tools: List[Dict[str, Any]], state: Dict[str, Any], conversation_history: List[Dict[str, Any]], st, encoding) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    # import pdb; pdb.set_trace()
    step_detail = {
            "purpose": st.get("purpose"),
            "required_state_keys": st.get("required_state_keys", []),
            "produces_state_keys": st.get("produces_state_keys", []),
            "step_query": st.get("step_query", ""),
            "state_details": st.get("state_details", {}),
            "attention_traps": st.get("attention_traps", [])
        }
    history_hint = (
        f"Step detail: " + json.dumps(step_detail, ensure_ascii=False)[:600]
    )
    history_text = build_history_text(conversation_history) if conversation_history else ""
    pack = {
        "tool_calls": [
            {"name": c.get("name"), "arguments": c.get("arguments", {})}
            for i, c in enumerate(tool_calls)
        ],
        "state": state,
        "history": history_text
        
    }
    instruction = "Execute and return results as a JSON list.\nINPUT:\n" + json.dumps(pack, ensure_ascii=False) + f"\nCurrent status hint: {history_hint}"
    convo = [
        {"role": "system", "content": TOOL_SIM_SYSTEM},
        {"role": "user", "content": instruction}
    ]
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    # import pdb; pdb.set_trace()
    out = chat_json(tokens)
    # import pdb; pdb.set_trace()
    reasoning = ""
    result = out['results']
    if "</think>" in out['results']:
        reasoning = out['results'].split("</think>")[0].strip()
        result = out['results'].split("</think>")[1].strip()
    try:
        doc = parse_json_str(result)
    except Exception as e:
        # import pdb; pdb.set_trace()
        print(f"[警告] Step  解析失败: {e}")
        doc = None
    ok, err = validate_tool_sim_json(doc or {}, len(tool_calls))
    if ok:
        # Return the list of tool results directly
        if isinstance(doc, list):
            return doc, reasoning
        else:
            # Convert old format to new format
            return [{"name": r["name"], "results": r.get("content", {})} for r in doc["tool_results"]], reasoning
    return None, ""

def run_agent_tool_calls(agent_system_template: str,
                        tool_catalog: str,
                        scenario: str,
                        domain: str,
                        topic: str,
                        user_msg: str,
                        world_state: Dict[str, Any],
                        allowed_tool_names: List[str],
                        conversation_history: List[Dict[str, Any]],
                        expected_tools: List[str] = None,
                        inject_plan_hint: str = None,
                        encoding=None,
                        tools: List[Dict[str, Any]] = None,
                        st: Dict[str, Any] = None,
                        suggested_tools_detail: List[Dict[str, Any]] = None,
                        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    运行agent tool calls，使用multi-step逻辑（最多7次循环）
    返回: (结果字典, 汇总消息列表)
    """
    # import pdb; pdb.set_trace()
    # Add plan hint if injected (after retries failed)
    plan_hint_text = ""
    if inject_plan_hint:
        # import pdb; pdb.set_trace()
        plan_hint_text = (
            f"\n\n[Internal Context - for reasoning only]: {inject_plan_hint}\n"
            "This step's target function calls are shown above. "
            "Please reference them and produce the corresponding tool invocations accordingly.\n"
        )
    # import pdb; pdb.set_trace()
    history_summary = create_history_summary(conversation_history, plan_hint_text)
    # 准备工具
    # import pdb; pdb.set_trace()
    convo = [
        {"role": "system", "content": gen_syten_message_qwen(tools)},
        *history_summary,
    ]
    if convo[-1]['role'] == "user":
        convo[-1]['content'] += user_msg
    else:
        convo.append({"role": "user", "content": user_msg})
    
    # 汇总消息列表
    all_messages = []
    
    # 构建 expected_tool_response 映射：tool_name -> expected_response
    expected_responses_map = {}
    if suggested_tools_detail:
        for t in suggested_tools_detail:
            if isinstance(t, dict) and "name" in t and "expected_tool_response" in t:
                expected_responses_map[t["name"]] = t["expected_tool_response"]
    
    # 初始化tokens
    tokens = encoding.apply_chat_template(convo, add_generation_prompt=True, thinking=True)
    
    # multi-step循环，最多7次
    max_loops = 10
    loop_count = 0
    all_tool_calls = []  # 收集所有的tool calls
    
    while loop_count < max_loops:
        loop_count += 1
        
        # 调用模型
        # import pdb; pdb.set_trace()
        out = chat_json(tokens)
        response_text = out.get("results", "")
        # import pdb; pdb.set_trace()
        
        # 1. 提取tool calls
        FC_info = extract_tool_calls(response_text)
        
        # 2. 如果没有tool call，提取summary并退出
        if len(FC_info) == 0:
            reasoning_content = ""
            summary_content = response_text
            if "</think>" in response_text:
                reasoning_content = response_text.split("</think>")[0].strip()
                summary_content = response_text.split("</think>")[1].strip()
            if summary_content:
                all_messages.append({
                    "role": "assistant",
                    "content": summary_content,
                    "reasoning": reasoning_content
                })
            break
        
        # 3. 有tool calls，进行处理
        # 3.1 添加tool_call消息 分开添加的话 reasoning_content很奇怪
        reasoning_content = ""
        if "</think>" in response_text:
            reasoning_content = response_text.split("</think>")[0].strip()
            response_text = response_text.split("</think>")[1].strip()
        all_tool_calls.extend(FC_info)
        all_messages.append({
                "role": "tool_call",
                "content": FC_info,
                "reasoning": reasoning_content
            })
        add_tokens_tool_call = []
        add_tokens_tool_call.append(
            {"role": "tool_call", "content": '\n'.join([f'<tool_call>\n{json.dumps(fc, ensure_ascii=False)}\n</tool_call>' for fc in FC_info])}
        )
        # import pdb; pdb.set_trace()
        tool_call_input_ids = encoding.apply_chat_template(add_tokens_tool_call, add_generation_prompt=True, thinking=True, chat_template=qwen_tool_call_template)
        tokens += tool_call_input_ids
        # 3.2 构建tool response：优先使用expected_tool_response，未匹配的退回模型模拟
        # import pdb; pdb.s et_trace()
        if expected_responses_map:
            temp_tool_response, temp_tool_reasoning = build_tool_responses_with_expected(
                FC_info, expected_responses_map, tools, world_state, conversation_history, st, encoding
            )
        else:
            temp_tool_response, temp_tool_reasoning = simulate_single_tool_call(
                FC_info, tools, world_state, conversation_history, st, encoding
            )
        
        if temp_tool_response is None:
            # 模拟失败，移除已添加的孤立 tool_call，保持 tool_call/tool_response 成对
            all_messages.pop()
            all_tool_calls = all_tool_calls[:-len(FC_info)]
            break
        
        # 3.3 添加tool_response消息
        all_messages.append({
                "role": "tool_response", 
                "content": temp_tool_response,
                "reasoning": temp_tool_reasoning  # tool_response的reasoning为对应的reasoning
            })
        
        # 3.4 将tool response添加到tokens以便继续
        # FIXME 关于多个工具的问题
        add_tokens = []
        # assert len(FC_info) == len(temp_tool_response)
        for tr in temp_tool_response:
            add_tokens.append({"role": "tool", "content": "<tool_response>" + json.dumps(tr) + "</tool_response>"})
        tool_response_input_ids = encoding.apply_chat_template(add_tokens, add_generation_prompt=True, thinking=True, chat_template=qwen_tool_response_template)
        tokens += tool_response_input_ids
    # import pdb; pdb.set_trace()
    # 验证tool calls是否匹配预期
    if expected_tools and all_tool_calls:
        # 提取所有调用的工具名称
        called_tools = [tc['name'] for tc in all_tool_calls]
        called_set = set(called_tools)
        expected_set = set(expected_tools)
        
        # 检查是否匹配
        if len(called_set) > len(expected_set) + 3:
            matches_plan = False
            too_many_tools = True
        else:
            matches_plan = expected_set.issubset(called_set)
            too_many_tools = False
        
        result = {
            "tool_calls": all_tool_calls,
            "matches_plan": matches_plan,
            "too_many_tools": too_many_tools
        }
    else:
        # 如果有tool calls就返回tool_calls，否则返回assistant
        if all_tool_calls:
            result = {
                "tool_calls": all_tool_calls,
                "matches_plan": True,
                "too_many_tools": False
            }
        else:
            assistant_msgs = [msg for msg in all_messages if msg['role'] == 'assistant']
            result = {
                "assistant": assistant_msgs[0]['content'] if assistant_msgs else "",
                "matches_plan": False,
                "too_many_tools": False
            }
    
    return result, all_messages




def run_agent_tool_calls_reflection(agent_system_template: str,
                        tool_catalog: str,
                        scenario: str,
                        domain: str,
                        topic: str,
                        user_msg: str,
                        world_state: Dict[str, Any],
                        allowed_tool_names: List[str],
                        conversation_history: List[Dict[str, Any]],
                        expected_tools: List[str] = None,
                        inject_plan_hint: str = None,
                        encoding=None,
                        tools: List[Dict[str, Any]] = None,
                        st: Dict[str, Any] = None,
                        suggested_tools_detail: List[Dict[str, Any]] = None,
                        agent_out=None,
                        agent_messages=None,
                        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    当 run_agent_tool_calls 未能覆盖全部 expected_tools 时，
    通过 hint 引导 user sim 生成新一轮对话，补齐缺失的工具调用。
    返回: (merged_agent_out, reflection_messages)
    """
    if not agent_out:
        return {}, agent_messages or []

    if agent_out.get("too_many_tools", False):
        return agent_out, agent_messages or []

    if agent_out.get("matches_plan", False):
        return agent_out, agent_messages or []

    # 计算缺失的工具
    actual_tools = [tc['name'] for tc in agent_out.get("tool_calls", [])]
    filtered_expected_tools = [t for t in (expected_tools or []) if t not in actual_tools]
    expected_tools_str = ", ".join(filtered_expected_tools) if filtered_expected_tools else "No specified tools"
    actual_tools_str = ", ".join(actual_tools) if actual_tools else "No tool calls"

    # 只保留缺失工具对应的 suggested_tools_detail
    filtered_suggested_tools_detail = None
    if suggested_tools_detail:
        filtered_suggested_tools_detail = [
            t for t in suggested_tools_detail
            if (t["name"] if isinstance(t, dict) else t) not in actual_tools
        ]

    # 构建 reflection hint
    reflection_hint = (
        f"Current step purpose: {st.get('purpose', '')}\n"
        f"Expected tools not yet called: {expected_tools_str}\n"
        f"Actually called tools: {actual_tools_str}\n"
        f"Tool matching status: Not matched - missing some expected tools.\n"
        f"Please generate a more accurate query to help the assistant call the missing tools."
    )

    try:
        new_user_msg, new_user_reasoning = run_user_query_for_step(
            USER_SIM_SYSTEM,
            tool_catalog,
            scenario,
            domain,
            topic,
            reflection_hint,
            conversation_history + (agent_messages or []),
            encoding
        )

        if not new_user_msg or len(new_user_msg.strip()) < 3:
            new_user_msg = user_msg

        # 以合并后的历史作为上下文开启新一轮对话
        new_agent_out, new_agent_messages = run_agent_tool_calls(
            agent_system_template,
            tool_catalog,
            scenario,
            domain,
            topic,
            new_user_msg,
            world_state,
            allowed_tool_names,
            conversation_history + (agent_messages or []),
            expected_tools=filtered_expected_tools,
            inject_plan_hint=inject_plan_hint,
            encoding=encoding,
            tools=tools,
            st=st,
            suggested_tools_detail=filtered_suggested_tools_detail,
        )

        # 合并两轮的 tool_calls
        merged_agent_out = new_agent_out or {}
        prev_calls = agent_out.get("tool_calls", []) if isinstance(agent_out, dict) else []
        new_calls = new_agent_out.get("tool_calls", []) if isinstance(new_agent_out, dict) else []
        merged_calls = prev_calls + new_calls
        if isinstance(merged_agent_out, dict):
            merged_agent_out["tool_calls"] = merged_calls

        # 用合并后的 tool_calls 重新判断 matches_plan
        if expected_tools:
            called_set = set(tc['name'] for tc in merged_calls)
            expected_set = set(expected_tools)
            if len(called_set) > len(expected_set) + 3:
                merged_agent_out["matches_plan"] = False
                merged_agent_out["too_many_tools"] = True
            else:
                merged_agent_out["matches_plan"] = expected_set.issubset(called_set)
                merged_agent_out["too_many_tools"] = False

        reflection_messages = [{
            "role": "user",
            "content": new_user_msg,
            "reasoning": new_user_reasoning
        }] + (new_agent_messages or [])

        return merged_agent_out, reflection_messages

    except Exception as e:
        print(f"[Reflection] Error: {e}")
        return agent_out or {}, agent_messages or []


def run_from_planning(planning_data: List[Dict[str, Any]], out_dir: str, data_index: int, round_index: int) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    从已有的planning数据生成交互对话
    planning_data: 包含planning信息的列表，每个元素是一个字典
    返回: (输出目录路径, 对话数据列表) 或 (None, None)
    """
    # import pdb; pdb.set_trace()
    encoding = AutoTokenizer.from_pretrained("/home/dongbingcheng/Agent_FC/ckpt/Qwen3-32B")
    # import pdb; pdb.set_trace()
    # 提取第一个字典（假设每个子列表只有一个字典）
    if not planning_data:
        return None, None
    # import pdb; pdb.set_trace()
    if 'system_info' in planning_data:
        system_info = planning_data['system_info']
    else:
        system_info = planning_data[0]
    
    # 提取系统信息
    scenario = system_info.get('scenario', '')
    domain = system_info.get('domain', '')
    topic = system_info.get('topic', '')
    tools = system_info.get('tools', [])
    # import pdb; pdb.set_trace()
    tools = [temp['function'] for temp in tools]
    plan_steps = system_info.get('planning', [])
    
    # 创建输出目录
    topic_slug = topic.replace("/", "_")
    tdir = os.path.join(out_dir, f"{topic_slug}_data{data_index}", f"round_{round_index}")
    os.makedirs(tdir, exist_ok=True)
    
    # 如果没有planning或tools，标记为bad case
    if not plan_steps or not tools:
        reason = "missing_planning_or_tools"
        mark_bad_case(tdir, reason)
        return None, None
    
    # 继承已有的 conv 和 statistics，从上次中断处续写
    try:
        existing_stats = planning_data.get('statistics', {})
        existing_steps_stats = list(existing_stats.get('steps_stats', []))
    except:
        existing_stats = {}
        existing_steps_stats = []

    # conv 只保留成功步骤的对话；steps_stats 末尾可能有一条 success=False 的失败记录
    # 去掉末尾失败记录，让该步重新生成
    if existing_steps_stats and not existing_steps_stats[-1].get('success', False):
        existing_steps_stats = existing_steps_stats[:-1]

    # 已成功完成的步骤数即为续写起点
    curr_step = len(existing_steps_stats)

    statistics = {
        "total_steps": len(plan_steps),
        "steps_stats": existing_steps_stats,
        "total_retries": existing_stats.get('total_retries', 0),
        "total_simulation_loops": existing_stats.get('total_simulation_loops', 0),
        "total_assist_queries": existing_stats.get('total_assist_queries', 0),
        "steps_with_retries": existing_stats.get('steps_with_retries', 0),
        "steps_with_simulation_loops": existing_stats.get('steps_with_simulation_loops', 0),
    }

    # 继承已有对话历史
    try:
        history_all = list(planning_data.get('conv', []))
    except:
        history_all = []
    world_state: Dict[str, Any] = {"cached": {}, "round_index": curr_step}
    error_occurred = False
    error_message = ""

    # 如果所有步骤已完成，直接保存并返回
    if curr_step >= len(plan_steps):
        conv_data = [{"system_info": system_info, "conv": history_all, "statistics": statistics}]
        dialogue_path = os.path.join(tdir, "dialogue.json")
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
        return tdir, conv_data

    try:
        for idx, st in enumerate(plan_steps[curr_step:]):
            world_state["round_index"] = idx + curr_step

            # 构建历史提示
            step_detail = {
                "purpose": st.get("purpose"),
                "suggested_tools": st.get("suggested_tools", []),
                "required_state_keys": st.get("required_state_keys", []),
                "produces_state_keys": st.get("produces_state_keys", []),
                "step_query": st.get("step_query", "")
            }
            history_hint = (
                f"Planning step {idx + 1 + curr_step}/{len(plan_steps)}:"
                f"Step detail: " + json.dumps(step_detail, ensure_ascii=False)[:600]
            )
            
            tool_catalog_text = render_tool_catalog(tools)
            
            # 生成用户查询
            try:
                user_msg, user_reasoning = run_user_query_for_step(
                    USER_SIM_SYSTEM,
                    tool_catalog_text,
                    scenario,
                    domain,
                    topic,
                    history_hint,
                    history_all,
                    encoding
                )
            except Exception as e:
                print(f"[警告] Step {idx+1} 用户查询生成失败: {e}")
                user_msg = st.get("step_query", "Please continue with the next step based on prior results.")
                user_reasoning = ""
            
            fallback_user = "Please continue with the next step based on prior results."
            if (not user_msg) or (len(user_msg.strip()) < 3) or (user_msg.strip() == fallback_user):
                if st.get("step_query"):
                    user_msg = st["step_query"].strip()
            # import pdb; pdb.set_trace()
            suggested_tools_detail = st.get("suggested_tools", [])
            # 提取工具名称列表（兼容 dict 格式和字符串格式）
            expected_tools = [
                t["name"] if isinstance(t, dict) else t
                for t in suggested_tools_detail
            ]
            agent_out = None
            max_retries = 1
            inject_plan_hint = None
            final_retry_attempt = 0
            
            # 当前step的统计信息
            step_stat = {
                "step_index": idx + 1 + curr_step,
                "retry_attempts": 0,
                "simulation_loops": 0,
                "assist_queries": 0,
                "expected_tools": expected_tools,
                "final_actual_tools": [],
                "success": False,
                "failure_reason": None
            }
            
            success_tool_call = 0
            reflection_retry_count = 0
            for retry_attempt in range(max_retries):
                # if retry_attempt > 0:
                #     import pdb; pdb.set_trace()
                final_retry_attempt = retry_attempt
                
                if retry_attempt == max_retries - 1:
                    plan_hint_parts = []
                    plan_hint_parts.append(f"This step's purpose: {st.get('purpose', '')}")
                    if expected_tools:
                        plan_hint_parts.append(f"Suggested tools for this step: {', '.join(expected_tools)}")
                    if st.get("state_details", {}).get("parameter_sources"):
                        plan_hint_parts.append(f"Parameter guidance: {st['state_details']['parameter_sources']}")
                    inject_plan_hint = " | ".join(plan_hint_parts)
                
                try:
                    # import pdb; pdb.set_trace()
                    agent_out, agent_messages = run_agent_tool_calls(
                        AGENT_SYSTEM_TEMPLATE,
                        tool_catalog_text,
                        scenario,
                        domain,
                        topic,
                        user_msg,
                        world_state,
                        [t["function"]["name"] for t in tools],
                        history_all,
                        expected_tools=expected_tools,
                        inject_plan_hint=inject_plan_hint,
                        encoding=encoding,
                        tools=tools,
                        st=st,
                        suggested_tools_detail=suggested_tools_detail,
                    )
                except Exception as e:
                    print(f"[警告] Step {idx + 1 + curr_step} retry {retry_attempt} agent调用失败: {e}")
                    if retry_attempt == max_retries - 1:
                        # 最后一次重试也失败了
                        step_stat["failure_reason"] = f"agent_call_error: {str(e)}"
                        break
                    continue
                
                if agent_out and agent_out.get("matches_plan"):
                    step_stat["retry_attempts"] = final_retry_attempt
                    actual_tool_names = [tc['name'] for tc in agent_out.get("tool_calls", [])]
                    step_stat["final_actual_tools"] = actual_tool_names
                    step_stat["success"] = True
                    
                    history_all.append({"role": "user", "content": user_msg, "reasoning": user_reasoning})
                    history_all.extend(agent_messages)
                    success_tool_call = 1
                    break
                else:
                    # 首轮未能覆盖全部 expected_tools，启动 reflection 补齐缺失工具
                    temp_conv = [{"role": "user", "content": user_msg, "reasoning": user_reasoning}]
                    temp_conv.extend(agent_messages)
                    new_agent_out_reflection, reflection_messages = run_agent_tool_calls_reflection(
                        AGENT_SYSTEM_TEMPLATE,
                        tool_catalog_text,
                        scenario,
                        domain,
                        topic,
                        user_msg,
                        world_state,
                        [t["function"]["name"] for t in tools],
                        history_all,
                        expected_tools=expected_tools,
                        inject_plan_hint=inject_plan_hint,
                        encoding=encoding,
                        tools=tools,
                        st=st,
                        suggested_tools_detail=suggested_tools_detail,
                        agent_out=agent_out,
                        agent_messages=temp_conv,
                    )

                    if new_agent_out_reflection and new_agent_out_reflection.get("matches_plan"):
                        # 将首轮对话 + reflection 轮对话都写入历史
                        history_all.append({"role": "user", "content": user_msg, "reasoning": user_reasoning})
                        history_all.extend(agent_messages)
                        history_all.extend(reflection_messages)
                        success_tool_call = 1
                        reflection_retry_count = 1
                        step_stat["success"] = True
                        actual_tool_names = [tc['name'] for tc in new_agent_out_reflection.get("tool_calls", [])]
                        step_stat["final_actual_tools"] = actual_tool_names
                        break
            
            # 如果失败，记录统计信息
            if success_tool_call == 0:
                step_stat["retry_attempts"] = final_retry_attempt
                if not step_stat["failure_reason"]:
                    step_stat["failure_reason"] = "max_retries_exceeded"
                statistics["steps_stats"].append(step_stat)
                statistics["total_retries"] += final_retry_attempt
                if final_retry_attempt > 0:
                    statistics["steps_with_retries"] += 1
                break
            
            # 成功的话，记录统计信息
            statistics["steps_stats"].append(step_stat)
            statistics["total_retries"] += step_stat["retry_attempts"]
            step_stat["reflection_retry_count"] = reflection_retry_count
            if step_stat["retry_attempts"] > 0:
                statistics["steps_with_retries"] += 1
    
    except Exception as e:
        error_occurred = True
        error_message = str(e)
        print(f"[错误] 处理过程中发生异常: {e}")
        statistics["error"] = error_message
        statistics["completed_steps"] = len(statistics["steps_stats"])
    
    # 无论是否出错，都保存已经生成的对话数据
    conv_data = [{
        "system_info": system_info,
        "conv": history_all,
        "statistics": statistics
    }]
    
    # 保存到dialogue.json
    try:
        dialogue_path = os.path.join(tdir, "dialogue.json")
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存统计信息到单独的文件
        statistics_path = os.path.join(tdir, "statistics.json")
        with open(statistics_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[错误] 保存文件失败: {e}")
    
    # 即使出错也返回数据
    return tdir, conv_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="from_planning", choices=["generate", "from_planning"], help="Mode: generate new planning or use existing planning")
    parser.add_argument("--planning_json", type=str, default="plan_0208/Base/res_6/valid_dialogues.json", help="Path to planning JSON file (for from_planning mode)")
    parser.add_argument("--out_dir", type=str, default="plan_0208/reflect/gen_5/merged_out")
    parser.add_argument("--tools_min", type=int, default=5, help="Min tools per scenario")
    parser.add_argument("--tools_max", type=int, default=9, help="Max tools per scenario")
    parser.add_argument("--rounds_min", type=int, default=3, help="Min conversation rounds")
    parser.add_argument("--rounds_max", type=int, default=6, help="Max conversation rounds")
    parser.add_argument("--repeated", type=int, default=4000, help="Repeat each category N times")
    parser.add_argument("--max_threads", type=int, default=512, help="Max parallel threads")
    parser.add_argument("--batch_size", type=int, default=512, help="Tasks per batch")
    parser.add_argument("--batches_dir", type=str, default="plan_0208/reflect/gen_5/batch", help="Directory to store per-batch metadata JSONs")
    parser.add_argument("--dialogues_jsonl", type=str, default="plan_0208/reflect/gen_5/merged_dialogues_modified.jsonl", help="Append all dialogues into a single JSONL file")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing planning data")
    parser.add_argument("--end_index", type=int, default=-1, help="End index for processing planning data (-1 means all)")
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"从 {args.planning_json} 读取planning数据...")
    
    # 使用流式读取处理大文件
    planning_file = args.planning_json if os.path.isabs(args.planning_json) else os.path.join("/home/dongbingcheng/Agent_FC", args.planning_json)
    
    if not os.path.exists(planning_file):
        print(f"错误: 找不到文件 {planning_file}")
        return
    
    # 逐行读取JSON数据
    print("正在读取planning数据...")
    with open(planning_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    planning_list = json.loads(content)
    print(f"总共读取到 {len(planning_list)} 条planning数据")
    
    # 处理索引范围
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index > 0 else len(planning_list)
    end_idx = min(end_idx, len(planning_list))
    
    print(f"处理范围: {start_idx} 到 {end_idx}")
    
    # 创建任务列表
    tasks = []
    for i in range(start_idx, end_idx):
        tasks.append({
            "planning_data": planning_list[i],
            "data_index": i,
            "round": 0
        })
    
    # 打开dialogues jsonl文件用于追加
    os.makedirs(os.path.dirname(os.path.abspath(args.dialogues_jsonl)) or '.', exist_ok=True)
    os.makedirs(args.batches_dir, exist_ok=True)
    
    completed = 0
    total_with_data = 0  # 统计有数据返回的任务数（即使不完全成功）
    have_batches = len(os.listdir(args.batches_dir))
    with open(args.dialogues_jsonl, 'a', encoding='utf-8') as jsonl_out:
        for i in range(0, len(tasks), args.batch_size):
            batch_idx = i // args.batch_size
            if batch_idx < have_batches:
                print(f"Skipping already processed batch {batch_idx + 1}")
                continue
            batch_tasks = tasks[i:i + args.batch_size]
            print(f"处理批次 {batch_idx + 1} (任务 {i} ~ {min(i + args.batch_size, len(tasks)) - 1}) ...")
            
            # 并发运行批次
            def _job(task_idx: int, task: Dict[str, Any]) -> Tuple[int, Optional[str], Optional[List[Dict[str, Any]]]]:
                try:
                    path, messages = run_from_planning(
                        task["planning_data"],
                        args.out_dir,
                        task["data_index"],
                        task["round"]
                    )
                    return task_idx, path, messages
                except Exception as e:
                    print(f"[错误] 任务#{task_idx} 失败: {e}")
                    return task_idx, None, None
            
            batch_results = []
            batch_messages_all = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as ex:
                future_map = {ex.submit(_job, j, batch_tasks[j]): j for j in range(len(batch_tasks))}
                for fut in tqdm(concurrent.futures.as_completed(future_map), total=len(future_map)):
                    task_idx = future_map[fut]
                    try:
                        idx, path, messages = fut.result()
                        batch_results.append((path, messages))
                    except Exception as e:
                        print(f"[错误] 收集任务#{task_idx}结果时出错: {e}")
                        batch_results.append((None, None))
            
            # 收集对话并写入JSONL，同时收集messages
            for j, (path, messages) in enumerate(batch_results):
                # 即使path为None，如果messages存在也要收集
                if messages:
                    batch_messages_all.append(messages)
                
                if not path:
                    continue
                
                try:
                    dlg_path = os.path.join(path, 'dialogue.json')
                    with open(dlg_path, 'r', encoding='utf-8') as f:
                        conv = json.load(f)
                    jsonl_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    completed += 1
                except Exception as e:
                    print(f"[警告] 无法追加任务 {i + j} 的对话: {e}")
                    # 即使文件读取失败，如果messages存在，也已经被收集了
                    continue
            
            jsonl_out.flush()
            
            # 保存batch messages到文件
            batch_messages_all = [temp for temp in batch_messages_all if temp is not None]
            batch_file = os.path.join(args.batches_dir, f"{batch_idx + 1}.json")
            with open(batch_file, 'w', encoding='utf-8') as bf:
                json.dump(batch_messages_all, bf, ensure_ascii=False, indent=4)
            
            # 统计当前批次的情况
            batch_success = sum(1 for path, _ in batch_results if path is not None)
            batch_has_messages = sum(1 for _, messages in batch_results if messages is not None)
            batch_total = len(batch_results)
            total_with_data += batch_has_messages
            
            print(f"批次 {batch_idx + 1} 完成: 成功 {batch_success}/{batch_total}, "
                    f"有数据 {batch_has_messages}/{batch_total}, "
                    f"总进度 {completed}/{len(tasks)}, "
                    f"保存到 {batch_file}")
    
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  完全成功: {completed}/{len(tasks)} 个任务")
    print(f"  有数据返回: {total_with_data}/{len(tasks)} 个任务（包括部分成功）")
    print(f"  输出目录: {args.out_dir}")
    print(f"  Batch文件目录: {args.batches_dir}")
    print(f"  JSONL文件: {args.dialogues_jsonl}")
    print(f"{'='*60}")
        



if __name__ == "__main__":
    main()

"""
python gen_new_plan/planning2interaction.py
"""



