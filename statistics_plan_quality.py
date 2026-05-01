"""
Planning质量统计工具
仿照validate_plan_quality函数，用于批量统计生成的planning质量
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import sys


def validate_plan_quality(plan: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    验证生成的plan是否满足质量要求
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
        "intra_step_dependencies": 0  # 统计步骤内依赖
    }
    
    produced_keys = set()
    
    for step_idx, step in enumerate(plan):
        # 统计多工具步骤
        suggested_tools = step.get("suggested_tools", [])
        if len(suggested_tools) >= 2:
            metrics["multi_tool_steps"] += 1
        
        # 统计包含attention traps的步骤
        if step.get("attention_traps") and len(step["attention_traps"]) > 0:
            metrics["steps_with_traps"] += 1
        
        # 统计包含详细state信息的步骤
        state_details = step.get("state_details", {})
        if (state_details.get("pre_conditions") or 
            state_details.get("post_conditions") or 
            state_details.get("concrete_values") or 
            state_details.get("parameter_sources")):
            metrics["steps_with_state_details"] += 1
        
        # 追踪state keys
        produces = step.get("produces_state_keys", [])
        requires = step.get("required_state_keys", [])
        
        for key in produces:
            # 处理key可能是字典的情况，转换为可哈希的字符串
            if isinstance(key, dict):
                key_str = json.dumps(key, sort_keys=True, ensure_ascii=False)
            else:
                key_str = str(key)
            produced_keys.add(key_str)
            metrics["total_state_keys"] += 1
        
        # 检查所需的keys是否在之前的步骤中产生
        for key in requires:
            # 同样处理key可能是字典的情况
            if isinstance(key, dict):
                key_str = json.dumps(key, sort_keys=True, ensure_ascii=False)
            else:
                key_str = str(key)
            if key_str in produced_keys:
                metrics["state_key_reuse"] += 1
        
        # 检查步骤内依赖（同一步骤内的工具相互依赖）
        state_details = step.get("state_details", {})
        param_sources = state_details.get("parameter_sources", {})
        concrete_vals = state_details.get("concrete_values", {})
        
        # 检查可疑模式，表示同步骤依赖
        suspicious_patterns = [
            "<to be filled",
            "from previous tool in this step",
            "output of earlier tool call",
            "result from first tool",
            "from tool call above"
        ]
        
        # 检查parameter_sources
        if isinstance(param_sources, dict):
            for param_name, source_desc in param_sources.items():
                if isinstance(source_desc, str):
                    source_lower = source_desc.lower()
                    if any(pattern in source_lower for pattern in suspicious_patterns):
                        # 检查是否指向同一步骤
                        if f"step {step_idx+1}" in source_lower or "this step" in source_lower or "same step" in source_lower:
                            metrics["intra_step_dependencies"] += 1
                            break
        
        # 检查concrete_values  
        if isinstance(concrete_vals, dict):
            for val_name, val in concrete_vals.items():
                if isinstance(val, str) and any(pattern in val.lower() for pattern in suspicious_patterns):
                    metrics["intra_step_dependencies"] += 1
                    break
    
    # 质量检查
    multi_tool_ratio = metrics["multi_tool_steps"] / metrics["total_steps"]
    state_detail_ratio = metrics["steps_with_state_details"] / metrics["total_steps"]
    
    # 至少25%应该使用多工具
    if multi_tool_ratio < 0.25:
        return False, f"Too few multi-tool steps: {metrics['multi_tool_steps']}/{metrics['total_steps']} ({multi_tool_ratio:.1%})", metrics
    
    # 至少40%应该有详细的state信息
    if state_detail_ratio < 0.4:
        return False, f"Too few steps with state details: {metrics['steps_with_state_details']}/{metrics['total_steps']} ({state_detail_ratio:.1%})", metrics
    
    # 应该至少有1个attention trap
    if metrics["steps_with_traps"] < 1:
        return False, "No attention traps found", metrics
    
    # 应该有状态传播（keys被重用）
    if metrics["total_steps"] > 2 and metrics["state_key_reuse"] < 1:
        return False, "No state key reuse detected - steps may not be connected", metrics
    
    # 不应该有步骤内依赖
    if metrics["intra_step_dependencies"] > 0:
        return False, f"Found {metrics['intra_step_dependencies']} step(s) with intra-step tool dependencies - tools in same step should not depend on each other", metrics
    
    return True, "Plan quality validated", metrics


class PlanQualityStatistics:
    """Planning质量统计器"""
    
    def __init__(self):
        self.total_plans = 0
        self.valid_plans = 0
        self.invalid_plans = 0
        self.invalid_reasons = defaultdict(int)
        
        # 聚合指标
        self.aggregated_metrics = {
            "total_steps": [],
            "multi_tool_steps": [],
            "steps_with_traps": [],
            "steps_with_state_details": [],
            "total_state_keys": [],
            "state_key_reuse": [],
            "intra_step_dependencies": []
        }
        
        # 比例指标
        self.ratio_metrics = {
            "multi_tool_ratio": [],
            "trap_ratio": [],
            "state_detail_ratio": [],
            "state_reuse_ratio": []
        }
    
    def add_plan(self, plan: List[Dict[str, Any]], plan_id: str = None) -> Dict[str, Any]:
        """添加一个plan进行统计"""
        self.total_plans += 1
        
        is_valid, reason, metrics = validate_plan_quality(plan)
        
        result = {
            "plan_id": plan_id,
            "is_valid": is_valid,
            "reason": reason,
            "metrics": metrics
        }
        
        if is_valid:
            self.valid_plans += 1
        else:
            self.invalid_plans += 1
            # 提取失败的主要原因
            reason_key = reason.split(":")[0] if ":" in reason else reason
            self.invalid_reasons[reason_key] += 1
        
        # 收集指标
        if metrics:
            for key in self.aggregated_metrics:
                if key in metrics:
                    self.aggregated_metrics[key].append(metrics[key])
            
            # 计算比例指标
            if metrics["total_steps"] > 0:
                self.ratio_metrics["multi_tool_ratio"].append(
                    metrics["multi_tool_steps"] / metrics["total_steps"]
                )
                self.ratio_metrics["trap_ratio"].append(
                    metrics["steps_with_traps"] / metrics["total_steps"]
                )
                self.ratio_metrics["state_detail_ratio"].append(
                    metrics["steps_with_state_details"] / metrics["total_steps"]
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
        
        def safe_min(lst):
            return min(lst) if lst else 0
        
        def safe_max(lst):
            return max(lst) if lst else 0
        
        summary = {
            "总计": {
                "总plan数": self.total_plans,
                "有效plan数": self.valid_plans,
                "无效plan数": self.invalid_plans,
                "有效率": f"{self.valid_plans / self.total_plans * 100:.2f}%" if self.total_plans > 0 else "0%"
            },
            "无效原因分布": dict(self.invalid_reasons),
            "步骤统计": {
                "平均步骤数": f"{safe_avg(self.aggregated_metrics['total_steps']):.2f}",
                "最小步骤数": safe_min(self.aggregated_metrics['total_steps']),
                "最大步骤数": safe_max(self.aggregated_metrics['total_steps']),
            },
            "多工具使用": {
                "平均多工具步骤数": f"{safe_avg(self.aggregated_metrics['multi_tool_steps']):.2f}",
                "平均多工具比例": f"{safe_avg(self.ratio_metrics['multi_tool_ratio']) * 100:.2f}%",
                "最小多工具比例": f"{safe_min(self.ratio_metrics['multi_tool_ratio']) * 100:.2f}%",
                "最大多工具比例": f"{safe_max(self.ratio_metrics['multi_tool_ratio']) * 100:.2f}%",
            },
            "Attention Traps": {
                "平均trap数": f"{safe_avg(self.aggregated_metrics['steps_with_traps']):.2f}",
                "平均trap比例": f"{safe_avg(self.ratio_metrics['trap_ratio']) * 100:.2f}%",
            },
            "状态详情": {
                "平均详细状态步骤数": f"{safe_avg(self.aggregated_metrics['steps_with_state_details']):.2f}",
                "平均状态详情比例": f"{safe_avg(self.ratio_metrics['state_detail_ratio']) * 100:.2f}%",
            },
            "状态键管理": {
                "平均状态键总数": f"{safe_avg(self.aggregated_metrics['total_state_keys']):.2f}",
                "平均状态键重用次数": f"{safe_avg(self.aggregated_metrics['state_key_reuse']):.2f}",
                "平均状态键重用比例": f"{safe_avg(self.ratio_metrics['state_reuse_ratio']) * 100:.2f}%",
            },
            "依赖问题": {
                "平均步骤内依赖数": f"{safe_avg(self.aggregated_metrics['intra_step_dependencies']):.2f}",
                "有步骤内依赖的plan数": sum(1 for x in self.aggregated_metrics['intra_step_dependencies'] if x > 0),
            }
        }
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("Planning 质量统计报告")
        print("="*80)
        
        for section, data in summary.items():
            print(f"\n【{section}】")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path: str):
        """保存统计报告到文件"""
        summary = self.get_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n统计报告已保存到: {output_path}")


def load_plans_from_json(file_path: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """从JSON文件加载plans"""
    plans = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持多种数据格式
    if isinstance(data, list):
        # 直接是plan列表
        if data and isinstance(data[0], dict) and "step_number" in data[0]:
            plans.append(("plan_0", data))
        # 是包含多个plan的列表
        else:
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # 如果item包含plan字段
                    if "plan" in item:
                        plan_id = item.get("id", f"plan_{idx}")
                        plans.append((plan_id, item["plan"]))
                    # 如果item包含planning字段
                    elif "planning" in item:
                        plan_id = item.get("id", f"plan_{idx}")
                        plans.append((plan_id, item["planning"]))
                    else:
                        # 假设item本身是一个plan
                        plans.append((f"plan_{idx}", [item]))
    elif isinstance(data, dict):
        # 如果是字典，可能包含多个plans
        if "plans" in data:
            for idx, plan in enumerate(data["plans"]):
                plan_id = plan.get("id", f"plan_{idx}")
                plans.append((plan_id, plan.get("plan", plan.get("planning", plan))))
        else:
            # 假设整个dict是一个plan的包装
            plans.append(("plan_0", data.get("plan", data.get("planning", []))))
    
    return plans


def load_plans_from_jsonl(file_path: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """从JSONL文件加载plans"""
    plans = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 提取plan
                plan_id = data.get("id", f"plan_{idx}")
                
                if "plan" in data:
                    plans.append((plan_id, data["plan"]))
                elif "planning" in data:
                    plans.append((plan_id, data["planning"]))
                elif isinstance(data, list):
                    plans.append((plan_id, data))
                else:
                    # 尝试在results中查找
                    if "results" in data:
                        results = data["results"]
                        if isinstance(results, str):
                            results = json.loads(results)
                        if "plan" in results:
                            plans.append((plan_id, results["plan"]))
                        elif "planning" in results:
                            plans.append((plan_id, results["planning"]))
            except json.JSONDecodeError as e:
                print(f"警告: 第{idx+1}行JSON解析失败: {e}")
                continue
    
    return plans


def main():
    parser = argparse.ArgumentParser(description="统计生成的Planning质量")
    parser.add_argument("input_file", type=str, help="输入文件路径 (支持.json或.jsonl格式)")
    parser.add_argument("--output", "-o", type=str, help="输出报告文件路径 (可选)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--show-invalid", action="store_true", help="显示无效的plan详情")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 加载plans
    print(f"正在加载plans从: {args.input_file}")
    
    if input_path.suffix == ".jsonl":
        plans = load_plans_from_jsonl(args.input_file)
    else:
        plans = load_plans_from_json(args.input_file)
    
    print(f"共加载 {len(plans)} 个plans\n")
    
    # 统计质量
    stats = PlanQualityStatistics()
    invalid_details = []
    
    for plan_id, plan in plans:
        result = stats.add_plan(plan, plan_id)
        
        if args.verbose:
            status = "✓ 有效" if result["is_valid"] else "✗ 无效"
            print(f"{plan_id}: {status} - {result['reason']}")
        
        if not result["is_valid"] and args.show_invalid:
            invalid_details.append(result)
    
    # 打印统计摘要
    stats.print_summary()
    
    # 显示无效plan详情
    if args.show_invalid and invalid_details:
        print("\n" + "="*80)
        print("无效Plan详情")
        print("="*80)
        for detail in invalid_details:
            print(f"\nPlan ID: {detail['plan_id']}")
            print(f"原因: {detail['reason']}")
            print(f"指标: {json.dumps(detail['metrics'], ensure_ascii=False, indent=2)}")
    
    # 保存报告
    if args.output:
        stats.save_report(args.output)


if __name__ == "__main__":
    main()

