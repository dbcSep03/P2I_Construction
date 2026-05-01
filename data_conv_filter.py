"""
过滤对话质量
"""
import json
from tqdm.auto import tqdm
import os 
import re


def data_static_ana(path_all,output_path):
    data_all = []
    info_all = {}
    info_all_planning_turn = {}
    info_all_reality_turn = {}
    save_success = []
    need_planning = []
    for path in tqdm(path_all, desc="Processing data", total=len(path_all)):
        name = os.listdir(path)
        for n in tqdm(name, desc="Processing data", total=len(name)):
            with open(f"{path}/{n}", 'r',encoding='utf-8') as f:
                data = json.load(f)
                data_all.extend(data)
    for d in tqdm(data_all, desc="Processing data", total=len(data_all)):
        d = d[0]

        statistics = d['statistics']
        step_all = statistics['total_steps']
        success_num = sum([1 for temp in statistics['steps_stats'] if temp['success']])
        
        if success_num == 0 or len(d['conv']) ==0:
            continue
        info_all[f'{success_num}/{step_all}'] = info_all.get(f'{success_num}/{step_all}', 0) + 1
        info_all_planning_turn[f'{step_all}'] = info_all_planning_turn.get(f'{step_all}', 0) + 1
        info_all_reality_turn[f"{success_num}"] = info_all_reality_turn.get(f"{success_num}", 0) + 1
        if success_num == step_all:
            save_success.append(d)
        else:
            if step_all> 3:
                need_planning.append(d)
    info_all =  sorted(info_all.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all:
        print(f"{k}: {v}")
    print("="*50)
    info_all_planning_turn =  sorted(info_all_planning_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_planning_turn:
        print(f"{k}: {v}")
    print("="*50)
    info_all_reality_turn =  sorted(info_all_reality_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_reality_turn:
        print(f"{k}: {v}")
    print("="*50)
    print(f"save_success: {len(save_success)}")
    print(f"need_planning: {len(need_planning)}")
    # with open(f"{output_path}/success_gpt_oss_1017.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success, f ,ensure_ascii=False, indent=4)
    # with open(f"{output_path}/success_gpt_oss_1017_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success[:20], f ,ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_gpt_oss_1017.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning, f ,ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_gpt_oss_1017_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning[:20], f ,ensure_ascii=False, indent=4)


def data_static_ana_qwen(path_all,output_path):
    data_all = []
    info_all = {}
    info_all_planning_turn = {}
    info_all_reality_turn = {}
    save_success = []
    need_planning = []
    for path in tqdm(path_all, desc="Processing data", total=len(path_all)):
        name = os.listdir(path)
        for n in tqdm(name, desc="Processing data", total=len(name)):
            with open(f"{path}/{n}", 'r',encoding='utf-8') as f:
                data = json.load(f)
                data_all.extend(data)
    for d in tqdm(data_all, desc="Processing data", total=len(data_all)):
        d = d[0]

        statistics = d['statistics']
        step_all = statistics['total_steps']
        success_num = sum([1 for temp in statistics['steps_stats'] if temp['success']])
        
        if success_num == 0 or len(d['conv']) ==0 or step_all == 1:
            continue
        info_all[f'{success_num}/{step_all}'] = info_all.get(f'{success_num}/{step_all}', 0) + 1
        info_all_planning_turn[f'{step_all}'] = info_all_planning_turn.get(f'{step_all}', 0) + 1
        info_all_reality_turn[f"{success_num}"] = info_all_reality_turn.get(f"{success_num}", 0) + 1
        if success_num == step_all:
            save_success.append(d)
        else:
            if step_all> 3:
                need_planning.append(d)
    info_all =  sorted(info_all.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all:
        print(f"{k}: {v}")
    print("="*50)
    info_all_planning_turn =  sorted(info_all_planning_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_planning_turn:
        print(f"{k}: {v}")
    print("="*50)
    info_all_reality_turn =  sorted(info_all_reality_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_reality_turn:
        print(f"{k}: {v}")
    print("="*50)
    print(f"save_success: {len(save_success)}")
    print(f"need_planning: {len(need_planning)}")

    with open(f"{output_path}/only_success_all.json", 'w', encoding='utf-8') as f:
        json.dump(save_success, f ,ensure_ascii=False, indent=4)
    with open(f"{output_path}/only_success_all_sample20.json", 'w', encoding='utf-8') as f:
        json.dump(save_success[:20], f ,ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning, f ,ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning[:20], f ,ensure_ascii=False, indent=4)


def _conv_has_tool_call_with_error(conv):
    """检查 conv 中是否存在 role 为 tool_call 且 is_error 为 True 的消息。"""
    for msg in conv:
        if msg.get("role") == "tool_call" and msg.get("is_error") is True:
            return True
        # content 可能是 list，其元素（或嵌套）中也可能有 is_error
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("is_error") is True:
                    return True
    return False


def data_conv_filter_error_gen(path_all, output_path):
    """
    仿照 data_static_ana_qwen：从 path_all 多个目录下两层遍历读取 json 文件。
    1. conv 中 role 为 reasoning 的删除
    2. 仅保留 conv 中存在 role 为 tool_call 且含 "is_error": true 的整条记录，否则整条删除
    3. 统计 info_all、info_all_planning_turn、info_all_reality_turn
    4. 按 success_all / need_planning_all 逻辑保存（success_num==step_all -> success_all，否则 step_all>3 -> need_planning_all）
    """
    os.makedirs(output_path, exist_ok=True)
    data_all = []
    for path in tqdm(path_all, desc="Reading paths", total=len(path_all)):
        name = os.listdir(path)
        for n in tqdm(name, desc="Reading files", total=len(name)):
            with open(f"{path}/{n}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_all.extend(data)

    info_all = {}
    info_all_planning_turn = {}
    info_all_reality_turn = {}
    save_success = []
    need_planning = []

    for item in tqdm(data_all, desc="Filtering data", total=len(data_all)):
        d = item[0]
        conv = d.get("conv", [])
        # 只保留至少有一条 tool_call 且 is_error 为 True 的记录
        if not _conv_has_tool_call_with_error(conv):
            continue
        # 删除 conv 中 role 为 reasoning 的消息
        new_conv = [m for m in conv if m.get("role") != "reasoning"]
        d = dict(d)
        d["conv"] = new_conv

        statistics = d['statistics']
        step_all = statistics['total_steps']
        success_num = sum([1 for temp in statistics['steps_stats'] if temp['success']])
        if success_num == 0 or len(d['conv']) == 0 or step_all == 1 or success_num == 1:
            continue

        info_all[f'{success_num}/{step_all}'] = info_all.get(f'{success_num}/{step_all}', 0) + 1
        info_all_planning_turn[f'{step_all}'] = info_all_planning_turn.get(f'{step_all}', 0) + 1
        info_all_reality_turn[f"{success_num}"] = info_all_reality_turn.get(f"{success_num}", 0) + 1
        if success_num == step_all:
            save_success.append(d)
        else:
            if step_all > 3:
                need_planning.append(d)

    info_all = sorted(info_all.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all:
        print(f"{k}: {v}")
    print("=" * 50)
    info_all_planning_turn = sorted(info_all_planning_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_planning_turn:
        print(f"{k}: {v}")
    print("=" * 50)
    info_all_reality_turn = sorted(info_all_reality_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_reality_turn:
        print(f"{k}: {v}")
    print("=" * 50)
    print(f"save_success: {len(save_success)}")
    print(f"need_planning: {len(need_planning)}")
    error_data_all = save_success + need_planning
    with open(f"{output_path}/error_data_all.json", 'w', encoding='utf-8') as f:
        json.dump(error_data_all, f, ensure_ascii=False, indent=4)
    with open(f"{output_path}/error_data_all_sample20.json", 'w', encoding='utf-8') as f:
        json.dump(error_data_all[:20], f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/success_all.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success, f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/success_all_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success[:20], f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning, f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning[:20], f, ensure_ascii=False, indent=4)


def data_conv_filter_reflect_gen(path_all, output_path):
    """
    处理 reflect/gen/batch 下的数据（如 plan_0208/reflect/gen/batch/1.json）。
    传入 path_all（目录列表），两层 tqdm 读取 json；
    只保留 statistics.total_retries != 0 的记录，total_retries 为 0 的过滤掉。
    统计 info_all_planning_turn、info_all_reality_turn，并按 success_all / need_planning_all 保存。
    """
    os.makedirs(output_path, exist_ok=True)
    data_all = []
    for path in tqdm(path_all, desc="Reading paths", total=len(path_all)):
        name = os.listdir(path)
        for n in tqdm(name, desc="Reading files", total=len(name)):
            with open(f"{path}/{n}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_all.extend(data)

    info_all = {}
    info_all_planning_turn = {}
    info_all_reality_turn = {}
    save_success = []
    need_planning = []

    for item in tqdm(data_all, desc="Filtering data", total=len(data_all)):
        d = item[0]
        statistics = d.get('statistics', {})
        # total_retries = statistics.get('total_retries', 0)
        total_retries = sum([temp.get('reflection_retry_count', 0) for temp in statistics.get('steps_stats', [])])
        if total_retries == 0:
            continue
        if len(d.get('conv', [])) == 0:
            continue

        step_all = statistics['total_steps']
        success_num = sum([1 for temp in statistics['steps_stats'] if temp['success']])
        if success_num == 0 or step_all == 1 or success_num == 1:
            continue
        conv = d.get('conv', [])
        role_all = [m.get('role', '') for m in conv]
        truncate_idx = None
        for idx, r in enumerate(role_all):
            if r == "tool_call":
                if idx + 1 >= len(role_all) or role_all[idx + 1] != "tool_response":
                    truncate_idx = idx
                    break
        if truncate_idx is not None:
            # 向前找到本轮的 user 消息，截断到该 user 之前（丢弃本轮）
            user_idx = truncate_idx
            for back_idx in range(truncate_idx - 1, -1, -1):
                if conv[back_idx].get('role') == 'user':
                    user_idx = back_idx
                    break
            d['conv'] = conv[:user_idx]
            success_num = sum(1 for m in d['conv'] if m.get('role') == 'user')
            if success_num <= 1:
                continue



        info_all[f'{success_num}/{step_all}'] = info_all.get(f'{success_num}/{step_all}', 0) + 1
        info_all_planning_turn[f'{step_all}'] = info_all_planning_turn.get(f'{step_all}', 0) + 1
        info_all_reality_turn[f"{success_num}"] = info_all_reality_turn.get(f"{success_num}", 0) + 1
        if success_num == step_all:
            save_success.append(d)
        else:
            if step_all > 3:
                need_planning.append(d)

    info_all = sorted(info_all.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all:
        print(f"{k}: {v}")
    print("=" * 50)
    info_all_planning_turn = sorted(info_all_planning_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_planning_turn:
        print(f"{k}: {v}")
    print("=" * 50)
    info_all_reality_turn = sorted(info_all_reality_turn.items(), key=lambda x: x[1], reverse=True)
    for k, v in info_all_reality_turn:
        print(f"{k}: {v}")
    print("=" * 50)
    print(f"save_success: {len(save_success)}")
    print(f"need_planning: {len(need_planning)}")
    reflect_data_all = save_success + need_planning
    with open(f"{output_path}/reflect_data_all.json", 'w', encoding='utf-8') as f:
        json.dump(reflect_data_all, f, ensure_ascii=False, indent=4)
    with open(f"{output_path}/reflect_data_all_sample20.json", 'w', encoding='utf-8') as f:
        json.dump(reflect_data_all[:20], f, ensure_ascii=False, indent=4)

    # with open(f"{output_path}/success_all.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success, f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/success_all_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(save_success[:20], f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning, f, ensure_ascii=False, indent=4)
    # with open(f"{output_path}/need_planning_all_sample20.json", 'w', encoding='utf-8') as f:
    #     json.dump(need_planning[:20], f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # data_static_ana(['gen_new_1023_oss_plan_80G_conv/batch'], 'data_res')

    # data_static_ana_qwen(['plan_0208/gen/batch','plan_0208/gen_1/batch', 'plan_0208/gen_2/batch'], '/home/dongbingcheng/Agent_FC/plan_0208/finnal_res')
    # data_static_ana_qwen(['plan_0208_40G/gen/batch', 'plan_0208_40G/gen_1/batch', 'plan_0208_40G/gen_2/batch'], '/home/dongbingcheng/Agent_FC/plan_0208_40G/finnal_res')

    # data_static_ana_qwen(['plan_0208_40G/gen/batch', 'plan_0208_40G/gen_1/batch', 'plan_0208_40G/gen_2/batch',
    #                        'plan_0208/gen/batch','plan_0208/gen_1/batch', 'plan_0208/gen_2/batch'], '/home/dongbingcheng/Agent_FC/plan_0208_final/raw_data')

    # data_conv_filter_error_gen(
    #     ['plan_0208_40G/ErrorResponse/error_gen/batch', 'plan_0208_40G/ErrorResponse/error_gen_1/batch'],
    #     'plan_0208_40G/ErrorResponse/gen_res'
    # )
    # data_conv_filter_error_gen(
    #     ['plan_0208/ErrorResponse/error_gen/batch', 'plan_0208/ErrorResponse/error_gen_1/batch'],
    #     'plan_0208/ErrorResponse/gen_res'
    # )

    # data_conv_filter_error_gen(
    #     ['plan_0208_40G/ErrorResponse/error_gen/batch', 'plan_0208_40G/ErrorResponse/error_gen_1/batch', 'plan_0208/ErrorResponse/error_gen/batch', 'plan_0208/ErrorResponse/error_gen_1/batch'],
    #     'plan_0208_final/raw_data'
    # )

    # reflect：只保留 total_retries != 0，统计并保存 success_all / need_planning_all
    # data_conv_filter_reflect_gen(
    #     ['plan_0208/reflect/gen/batch', 'plan_0208/reflect/gen_2/batch', 'plan_0208/reflect/gen_3/batch','plan_0208/reflect/gen_4/batch', 'plan_0208/reflect/gen_5/batch'],
    #     'plan_0208/reflect/res'
    # )
    # data_conv_filter_reflect_gen(
    #     ['plan_0208_40G/reflect/gen/batch', 'plan_0208_40G/reflect/gen_2/batch', 'plan_0208_40G/reflect/gen_3/batch', 'plan_0208_40G/reflect/gen_4/batch', 'plan_0208_40G/reflect/gen_5/batch'],
    #     'plan_0208_40G/reflect/res'
    # )
    data_conv_filter_reflect_gen(
        ['plan_0208/reflect/gen/batch', 'plan_0208/reflect/gen_2/batch', 'plan_0208/reflect/gen_3/batch', 'plan_0208/reflect/gen_4/batch','plan_0208/reflect/gen_5/batch',
         'plan_0208_40G/reflect/gen/batch', 'plan_0208_40G/reflect/gen_2/batch', 'plan_0208_40G/reflect/gen_3/batch', 'plan_0208_40G/reflect/gen_4/batch', 'plan_0208_40G/reflect/gen_5/batch'],
        'plan_0208_final/raw_data'
    )

