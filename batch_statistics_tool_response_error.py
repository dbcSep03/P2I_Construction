"""
批量统计batch目录下所有tool response error planning文件的质量
专门用于分析 gen_plan_tool_error_qwen.py 生成的数据。

每个 suggested_tool 的结构：
  普通工具（2字段）：
    expected_right_tool_call  — 正确的工具调用 {name, arguments}
    expected_tool_response    — 成功响应

  错误工具（4字段，整个plan共2-3个）：
    expected_error_tool_call  — 初始错误调用 {name, arguments (错误参数)}
    expected_error_response   — 错误响应 {error_type, http_status, message, details}
    expected_right_tool_call  — 修复后的正确调用 {name, arguments (正确参数)}
    expected_tool_response    — 成功响应
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import sys
from statistics_plan_quality import PlanQualityStatistics


# 12种合法的 error_type 枚举值
ALLOWED_ERROR_TYPES = {
    "MISSING_REQUIRED_FIELD",
    "TYPE_MISMATCH",
    "INVALID_FORMAT",
    "CONSTRAINT_VIOLATION",
    "DANGLING_REFERENCE",
    "STALE_VERSION",
    "DEPENDENCY_NOT_SATISFIED",
    "PERMISSION_DENIED",
    "NOT_FOUND",
    "RATE_LIMIT_EXCEEDED",
    "PRECONDITION_FAILED",
    "SERVICE_UNAVAILABLE",
}

# error_type 所属的层次（用于分层统计）
ERROR_TYPE_LAYER = {
    "MISSING_REQUIRED_FIELD":   "Signature",
    "TYPE_MISMATCH":            "Signature",
    "INVALID_FORMAT":           "Signature",
    "CONSTRAINT_VIOLATION":     "Signature",
    "DANGLING_REFERENCE":       "World-state",
    "STALE_VERSION":            "World-state",
    "DEPENDENCY_NOT_SATISFIED": "World-state",
    "PERMISSION_DENIED":        "Execution",
    "NOT_FOUND":                "Execution",
    "RATE_LIMIT_EXCEEDED":      "Execution",
    "PRECONDITION_FAILED":      "Execution",
    "SERVICE_UNAVAILABLE":      "Execution",
}


def validate_tool_error_plan_quality(plan: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    验证 gen_plan_tool_error_qwen.py 生成的 plan 的质量。

    每个 suggested_tool 的新结构：
      普通工具: expected_right_tool_call + expected_tool_response
      错误工具: expected_error_tool_call + expected_error_response
               + expected_right_tool_call + expected_tool_response

    返回: (is_valid, reason, metrics)
    """
    if not plan:
        return False, "Plan is empty", {}

    metrics = {
        "total_steps": len(plan),
        "multi_tool_steps": 0,
        "steps_with_traps": 0,
        "steps_with_state_details": 0,
        "total_state_keys": 0,
        "state_key_reuse": 0,
        "intra_step_dependencies": 0,
        # 新4字段格式的错误工具统计
        "error_tools_total": 0,         # 携带 expected_error_tool_call 的工具数（整个plan）
        "steps_with_error_tool": 0,     # 至少有一个错误工具的步骤数
        "normal_tools_total": 0,        # 仅有 expected_right_tool_call 的普通工具数
        "error_type_counts": {},        # {error_type: count}
        "unknown_error_types": 0,       # error_type 不在枚举中的数量
        "tools_missing_error_type": 0,  # expected_error_response 里没有 error_type 字段
        "tools_missing_right_call": 0,  # 缺少 expected_right_tool_call 的工具数
    }

    produced_keys = set()

    for step_idx, step in enumerate(plan):
        suggested_tools = step.get("suggested_tools", [])
        if not isinstance(suggested_tools, list):
            suggested_tools = []

        # 多工具步骤
        if len(suggested_tools) >= 2:
            metrics["multi_tool_steps"] += 1

        # attention traps
        if step.get("attention_traps") and len(step["attention_traps"]) > 0:
            metrics["steps_with_traps"] += 1

        # state_details 完整性
        state_details = step.get("state_details", {})
        if not isinstance(state_details, dict):
            state_details = {}
        if (state_details.get("pre_conditions") or
                state_details.get("post_conditions") or
                state_details.get("concrete_values") or
                state_details.get("parameter_sources")):
            metrics["steps_with_state_details"] += 1

        # state key 追踪
        produces = step.get("produces_state_keys", [])
        requires = step.get("required_state_keys", [])
        for key in produces:
            key_str = json.dumps(key, sort_keys=True, ensure_ascii=False) if isinstance(key, dict) else str(key)
            produced_keys.add(key_str)
            metrics["total_state_keys"] += 1
        for key in requires:
            key_str = json.dumps(key, sort_keys=True, ensure_ascii=False) if isinstance(key, dict) else str(key)
            if key_str in produced_keys:
                metrics["state_key_reuse"] += 1

        # 步骤内依赖检查
        param_sources = state_details.get("parameter_sources", {})
        concrete_vals = state_details.get("concrete_values", {})
        suspicious_patterns = [
            "<to be filled",
            "from previous tool in this step",
            "output of earlier tool call",
            "result from first tool",
            "from tool call above",
        ]
        if isinstance(param_sources, dict):
            for _, source_desc in param_sources.items():
                if isinstance(source_desc, str):
                    src_lower = source_desc.lower()
                    if any(p in src_lower for p in suspicious_patterns):
                        if f"step {step_idx+1}" in src_lower or "this step" in src_lower or "same step" in src_lower:
                            metrics["intra_step_dependencies"] += 1
                            break
        if isinstance(concrete_vals, dict):
            for _, val in concrete_vals.items():
                if isinstance(val, str) and any(p in val.lower() for p in suspicious_patterns):
                    metrics["intra_step_dependencies"] += 1
                    break

        # ── 核心：遍历 suggested_tools，按新4字段结构统计 ──
        step_has_error_tool = False
        for tool_item in suggested_tools:
            if not isinstance(tool_item, dict):
                continue

            has_error_call = "expected_error_tool_call" in tool_item
            has_right_call = "expected_right_tool_call" in tool_item

            if not has_right_call:
                metrics["tools_missing_right_call"] += 1

            if has_error_call:
                # 错误工具：统计 error_type
                metrics["error_tools_total"] += 1
                step_has_error_tool = True

                err = tool_item.get("expected_error_response", {})
                if not isinstance(err, dict):
                    metrics["tools_missing_error_type"] += 1
                    continue

                # 兼容旧字段名 error_code
                error_type = err.get("error_type") or err.get("error_code")
                if not error_type:
                    metrics["tools_missing_error_type"] += 1
                    continue

                if error_type not in ALLOWED_ERROR_TYPES:
                    metrics["unknown_error_types"] += 1
                    error_type = f"UNKNOWN:{error_type}"

                metrics["error_type_counts"][error_type] = (
                    metrics["error_type_counts"].get(error_type, 0) + 1
                )
            else:
                # 普通工具
                metrics["normal_tools_total"] += 1

        if step_has_error_tool:
            metrics["steps_with_error_tool"] += 1

    # ── 质量检查 ──
    total = metrics["total_steps"]
    multi_tool_ratio = metrics["multi_tool_steps"] / total
    state_detail_ratio = metrics["steps_with_state_details"] / total

    if multi_tool_ratio < 0.25:
        return False, f"Too few multi-tool steps: {metrics['multi_tool_steps']}/{total} ({multi_tool_ratio:.1%})", metrics

    if state_detail_ratio < 0.4:
        return False, f"Too few steps with state details: {metrics['steps_with_state_details']}/{total} ({state_detail_ratio:.1%})", metrics

    if metrics["steps_with_traps"] < 1:
        return False, "No attention traps found", metrics

    if total > 2 and metrics["state_key_reuse"] < 1:
        return False, "No state key reuse detected - steps may not be connected", metrics

    if metrics["intra_step_dependencies"] > 0:
        return False, f"Found {metrics['intra_step_dependencies']} step(s) with intra-step tool dependencies", metrics

    # 整个 plan 至少有 2 个完整的错误工具（带 expected_error_tool_call）
    if metrics["error_tools_total"] < 2:
        return False, (
            f"Insufficient error tools: need >=2 tools with expected_error_tool_call, "
            f"got {metrics['error_tools_total']}"
        ), metrics

    return True, "Plan quality validated", metrics


def extract_plans_from_batch_file(file_path: str) -> List[Tuple[str, List[Dict[str, Any]], Any]]:
    """
    从batch文件中提取planning数据
    batch文件格式: 是一个对话数组，每个对话包含消息，消息中可能有planning字段
    返回: [(plan_id, planning, dialogue), ...]
    """
    plans = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return plans
    
    file_name = Path(file_path).stem
    
    # 如果是数组，遍历每个对话
    if isinstance(data, list):
        for dialogue_idx, dialogue in enumerate(data):
            if isinstance(dialogue, list):
                # 对话是一个消息列表，查找包含planning字段的消息
                for msg_idx, msg in enumerate(dialogue):
                    if isinstance(msg, dict) and "planning" in msg:
                        planning = msg["planning"]
                        if isinstance(planning, list) and len(planning) > 0:
                            plan_id = f"{file_name}_d{dialogue_idx}_m{msg_idx}"
                            # 返回 (plan_id, planning, 整个对话)
                            plans.append((plan_id, planning, dialogue))
            
            elif isinstance(dialogue, dict):
                # 如果对话本身是字典，可能直接包含plan或planning
                if "plan" in dialogue:
                    plan_id = f"{file_name}_dialogue_{dialogue_idx}"
                    plans.append((plan_id, dialogue["plan"], dialogue))
                elif "planning" in dialogue:
                    plan_id = f"{file_name}_dialogue_{dialogue_idx}"
                    plans.append((plan_id, dialogue["planning"], dialogue))
    
    return plans


def normalize_plan_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化plan step，确保字段名称一致
    将step_index转换为step_number等
    """
    normalized = step.copy()
    
    # 统一step编号字段
    if "step_index" in normalized and "step_number" not in normalized:
        normalized["step_number"] = normalized["step_index"]
    
    return normalized


class ToolResponseErrorPlanQualityStatistics(PlanQualityStatistics):
    """Tool Response Error Planning 质量统计器（适配 gen_plan_tool_error_qwen.py 新格式）"""

    def __init__(self):
        super().__init__()

        # 新4字段格式的聚合指标（列表，用于求均值/求和）
        self.aggregated_metrics.update({
            "error_tools_total": [],
            "steps_with_error_tool": [],
            "normal_tools_total": [],
            "unknown_error_types": [],
            "tools_missing_error_type": [],
            "tools_missing_right_call": [],
        })

        # 全局 error_type 计数（跨所有 plan 汇总）
        self.global_error_type_counter: Counter = Counter()

    def add_plan(self, plan: List[Dict[str, Any]], plan_id: str = None) -> Dict[str, Any]:
        """添加一个 plan 进行统计"""
        self.total_plans += 1

        is_valid, reason, metrics = validate_tool_error_plan_quality(plan)

        result = {
            "plan_id": plan_id,
            "is_valid": is_valid,
            "reason": reason,
            "metrics": metrics,
        }

        if is_valid:
            self.valid_plans += 1
        else:
            self.invalid_plans += 1
            reason_key = reason.split(":")[0] if ":" in reason else reason
            self.invalid_reasons[reason_key] += 1

        if metrics:
            # 收集数值指标
            for key in self.aggregated_metrics:
                if key in metrics and not isinstance(metrics[key], dict):
                    self.aggregated_metrics[key].append(metrics[key])

            # 汇总 error_type 分布
            for etype, cnt in metrics.get("error_type_counts", {}).items():
                self.global_error_type_counter[etype] += cnt

            # 比例指标
            total = metrics["total_steps"]
            if total > 0:
                self.ratio_metrics["multi_tool_ratio"].append(
                    metrics["multi_tool_steps"] / total
                )
                self.ratio_metrics["trap_ratio"].append(
                    metrics["steps_with_traps"] / total
                )
                self.ratio_metrics["state_detail_ratio"].append(
                    metrics["steps_with_state_details"] / total
                )
                if metrics["total_state_keys"] > 0:
                    self.ratio_metrics["state_reuse_ratio"].append(
                        metrics["state_key_reuse"] / metrics["total_state_keys"]
                    )

        return result

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0

        summary = super().get_summary()

        # 错误工具基础统计
        summary["错误工具统计"] = {
            "平均每plan错误工具数(expected_error_tool_call)": f"{safe_avg(self.aggregated_metrics['error_tools_total']):.2f}",
            "平均每plan含error工具的步骤数":                  f"{safe_avg(self.aggregated_metrics['steps_with_error_tool']):.2f}",
            "平均每plan普通工具数":                           f"{safe_avg(self.aggregated_metrics['normal_tools_total']):.2f}",
            "未知error_type工具总数":                         sum(self.aggregated_metrics['unknown_error_types']),
            "缺少error_type字段的工具总数":                   sum(self.aggregated_metrics['tools_missing_error_type']),
            "缺少expected_right_tool_call的工具总数":         sum(self.aggregated_metrics['tools_missing_right_call']),
        }

        # error_type 分布（按层次分组）
        total_error_tools = sum(self.global_error_type_counter.values())
        layer_counts: Dict[str, int] = defaultdict(int)
        type_detail: Dict[str, str] = {}
        for etype, cnt in sorted(self.global_error_type_counter.items(), key=lambda x: -x[1]):
            layer = ERROR_TYPE_LAYER.get(etype, "Unknown")
            layer_counts[layer] += cnt
            pct = cnt / total_error_tools * 100 if total_error_tools else 0
            type_detail[etype] = f"{cnt} ({pct:.1f}%)"

        summary["error_type分布（全局）"] = {
            "总计错误工具数": total_error_tools,
            "按类型": type_detail,
            "按层次": {
                layer: f"{cnt} ({cnt/total_error_tools*100:.1f}%)" if total_error_tools else "0"
                for layer, cnt in sorted(layer_counts.items())
            },
        }

        return summary


def batch_analyze_directory(directory: str, output_dir: str = None, verbose: bool = False):
    """批量分析目录下的所有JSON文件"""
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"错误: 目录不存在: {directory}")
        return
    
    # 获取所有JSON文件
    json_files = sorted(dir_path.glob("*.json"))
    
    if not json_files:
        print(f"错误: 目录中没有找到JSON文件: {directory}")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    print("="*80)
    
    # 总体统计
    overall_stats = ToolResponseErrorPlanQualityStatistics()
    
    # 每个文件的统计
    file_stats = {}
    
    # 保存通过审核的plans（包含整个对话）
    valid_dialogues = []
    
    for file_path in json_files:
        file_name = file_path.name
        print(f"\n处理文件: {file_name}")
        
        # 提取plans
        plans = extract_plans_from_batch_file(str(file_path))
        
        if not plans:
            print(f"  警告: 未能从文件中提取planning数据")
            continue
        
        print(f"  提取到 {len(plans)} 个plans")
        
        # 创建文件级别的统计
        file_stat = ToolResponseErrorPlanQualityStatistics()
        
        # 分析每个plan
        for plan_id, plan, dialogue in plans:
            # 标准化plan steps
            normalized_plan = [normalize_plan_step(step) for step in plan]
            
            # 添加到统计
            result = overall_stats.add_plan(normalized_plan, plan_id)
            file_stat.add_plan(normalized_plan, plan_id)
            
            # 如果通过审核，保存整个对话
            if result["is_valid"]:
                valid_dialogues.append({
                    "plan_id": plan_id,
                    "dialogue": dialogue,  # 保存整个对话
                    "metrics": result["metrics"]
                })
            
            if verbose:
                status = "✓" if result["is_valid"] else "✗"
                print(f"    {status} {plan_id}: {result['reason']}")
        
        # 保存文件级别统计
        file_stats[file_name] = {
            "total_plans": file_stat.total_plans,
            "valid_plans": file_stat.valid_plans,
            "invalid_plans": file_stat.invalid_plans,
            "valid_rate": f"{file_stat.valid_plans / file_stat.total_plans * 100:.2f}%" if file_stat.total_plans > 0 else "0%"
        }
    
    # 打印总体统计
    print("\n" + "="*80)
    print("Tool Response Error Planning 质量统计报告")
    print("="*80)
    overall_stats.print_summary()

    # 单独打印 error_type 分布（便于一眼看到）
    print("\n" + "="*80)
    print("error_type 分布统计")
    print("="*80)
    total_err = sum(overall_stats.global_error_type_counter.values())
    if total_err == 0:
        print("  （未发现任何 expected_error_response）")
    else:
        # 按层次分组打印
        layer_order = ["Signature", "World-state", "Execution"]
        layer_types = {layer: [] for layer in layer_order}
        for etype in ALLOWED_ERROR_TYPES:
            layer = ERROR_TYPE_LAYER.get(etype, "Unknown")
            if layer in layer_types:
                layer_types[layer].append(etype)
        for layer in layer_order:
            layer_total = sum(overall_stats.global_error_type_counter.get(et, 0) for et in layer_types[layer])
            print(f"\n  [{layer}]  合计: {layer_total}  ({layer_total/total_err*100:.1f}%)")
            for etype in sorted(layer_types[layer]):
                cnt = overall_stats.global_error_type_counter.get(etype, 0)
                bar = "█" * min(cnt, 40)
                print(f"    {etype:<30} {cnt:>6}  ({cnt/total_err*100:5.1f}%)  {bar}")
        # 未知类型
        unknowns = {k: v for k, v in overall_stats.global_error_type_counter.items()
                    if k not in ALLOWED_ERROR_TYPES}
        if unknowns:
            print(f"\n  [Unknown / Out-of-enum]")
            for etype, cnt in sorted(unknowns.items(), key=lambda x: -x[1]):
                print(f"    {etype:<30} {cnt:>6}")

    # 打印通过审核的对话数量
    print("\n" + "="*80)
    print(f"✓ 成功通过审核的对话总数: {len(valid_dialogues)}")
    print("="*80)
    
    # 打印每个文件的统计摘要
    print("\n" + "="*80)
    print("各文件统计摘要")
    print("="*80)
    print(f"{'文件名':<20} {'总计':<10} {'有效':<10} {'无效':<10} {'有效率':<10}")
    print("-"*80)
    for file_name, stats in sorted(file_stats.items()):
        print(f"{file_name:<20} {stats['total_plans']:<10} {stats['valid_plans']:<10} {stats['invalid_plans']:<10} {stats['valid_rate']:<10}")
    
    # 保存报告
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存总体报告
        overall_report_path = output_path / "overall_report.json"
        overall_stats.save_report(str(overall_report_path))
        
        # 保存文件级别报告
        file_report_path = output_path / "file_statistics.json"
        with open(file_report_path, 'w', encoding='utf-8') as f:
            json.dump(file_stats, f, ensure_ascii=False, indent=2)
        print(f"\n文件统计已保存到: {file_report_path}")
        
        # 保存通过审核的对话（包含完整信息：plan_id, dialogue, metrics）
        valid_dialogues_detailed_path = output_path / "valid_dialogues_detailed.json"
        with open(valid_dialogues_detailed_path, 'w', encoding='utf-8') as f:
            json.dump(valid_dialogues, f, ensure_ascii=False, indent=2)
        print(f"有效对话（详细信息）已保存到: {valid_dialogues_detailed_path} (共 {len(valid_dialogues)} 个)")
        
        # 保存纯对话列表（仅包含对话内容，格式与原始文件一致）
        valid_dialogues_only = [item["dialogue"] for item in valid_dialogues]
        valid_dialogues_path = output_path / "valid_dialogues.json"
        with open(valid_dialogues_path, 'w', encoding='utf-8') as f:
            json.dump(valid_dialogues_only, f, ensure_ascii=False, indent=2)
        print(f"有效对话（纯净列表）已保存到: {valid_dialogues_path}")
        
        # 另外保存一个JSONL格式（每行一个对话，方便后续处理）
        valid_dialogues_jsonl_path = output_path / "valid_dialogues.jsonl"
        with open(valid_dialogues_jsonl_path, 'w', encoding='utf-8') as f:
            for dialogue_item in valid_dialogues_only:
                f.write(json.dumps(dialogue_item, ensure_ascii=False) + '\n')
        print(f"有效对话 (JSONL格式) 已保存到: {valid_dialogues_jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="批量统计 gen_plan_tool_error_qwen.py 生成的 batch 目录下所有 planning 文件的质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析整个batch目录
  python batch_statistics_tool_response_error.py /home/dongbingcheng/Agent_FC/plan_0208/ErrorResponse/batch

  # 显示详细信息并保存报告
  python batch_statistics_tool_response_error.py /home/dongbingcheng/Agent_FC/plan_0208/ErrorResponse/batch -v -o ./reports
        """
    )
    
    parser.add_argument("directory", type=str, help="batch目录路径")
    parser.add_argument("--output", "-o", type=str, help="输出报告目录路径 (可选)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 批量分析
    batch_analyze_directory(args.directory, args.output, args.verbose)


if __name__ == "__main__":
    main()

"""
使用说明：
# 分析并保存包含有效 error response planning 的对话
python gen_plan_0208/batch_statistics_tool_response_error.py plan_0208/ErrorResponse/plan_1/batch -o plan_0208/ErrorResponse/res_1
python gen_plan_0208/batch_statistics_tool_response_error.py plan_0208_40G/ErrorResponse/plan_1/batch -o plan_0208_40G/ErrorResponse/res_1
# 运行后会在 ./reports 目录下生成：
# - overall_report.json              (总体统计报告，含 error_type 分布)
# - file_statistics.json             (各文件统计)
# - valid_dialogues_detailed.json    (包含 plan_id 和 metrics 的详细信息)
# - valid_dialogues.json             (仅包含对话的纯净列表)
# - valid_dialogues.jsonl            (JSONL 格式，每行一个对话)
"""
