#!/usr/bin/env python3
"""
学生自测脚本 - 离线评测Prompt代码的各项得分

用法:
    python self_test.py --data-dir . --code my_prompt.py --api-key YOUR_API_KEY
    python self_test.py --data-dir . --code my_prompt.py --api-key YOUR_API_KEY --type eval
    python self_test.py --data-dir . --code my_prompt.py --api-key YOUR_API_KEY --limit 10

参数:
    --data-dir  数据集目录(包含train.json, test_input.json等), 默认当前目录
    --code      学生编写的Prompt代码文件(必须包含generate_prompt(ctx)函数)
    --api-key   智谱AI的API Key
    --type      评测类型: test(测试集) 或 eval(评分集), 默认test
    --limit     只评测前N个样本(可选, 用于快速调试)
    --mock      使用随机预测代替API调用(不需要API Key, 用于调试代码逻辑)
"""

import json
import math
import random
import re
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# ============================================================
# PromptContext 类 - 包含所有ctx函数的完整实现
# ============================================================

class PromptContext:
    """Prompt生成上下文，提供给学生使用的数据和工具"""

    def __init__(self, user_history: List[Dict], target_movie: Dict,
                 movies_info: Dict, all_users_history: Optional[List[List[Dict]]] = None):
        self.user_history = user_history
        self.target_movie = target_movie
        self.movies_info = movies_info
        self.all_users_history = all_users_history or []

    def get_history_sample(self, n: int = 5, strategy: str = 'recent') -> List[Dict]:
        """
        从用户历史中采样

        Args:
            n: 采样数量
            strategy: 采样策略 - 'recent'(最近), 'random'(随机), 'highest'(最高分), 'lowest'(最低分)

        Returns:
            采样的历史记录列表
        """
        if not self.user_history:
            return []

        n = min(n, len(self.user_history))

        if strategy == 'recent':
            return self.user_history[-n:]
        elif strategy == 'random':
            return random.sample(self.user_history, n)
        elif strategy == 'highest':
            sorted_history = sorted(self.user_history, key=lambda x: x.get('rating', 0), reverse=True)
            return sorted_history[:n]
        elif strategy == 'lowest':
            sorted_history = sorted(self.user_history, key=lambda x: x.get('rating', 0))
            return sorted_history[:n]
        else:
            return self.user_history[-n:]

    def get_similar_movies(self, n: int = 5) -> List[Dict]:
        """
        获取与目标电影相似的电影（基于标签）

        Args:
            n: 返回数量

        Returns:
            用户评价过的相似电影列表
        """
        if not self.user_history or not self.target_movie:
            return []

        target_tags = set(self.target_movie.get('tags', '').split(', '))

        scored_history = []
        for item in self.user_history:
            movie_id = item.get('movie_id')
            movie_info = self.movies_info.get(movie_id, {})
            movie_tags = set(movie_info.get('tags', '').split(', '))
            similarity = len(target_tags & movie_tags)
            scored_history.append((similarity, item))

        scored_history.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_history[:n]]

    def get_random_user_history(self, n: int = 5) -> List[Dict]:
        """
        从其他用户的历史中随机采样（用于跨用户示例）

        Args:
            n: 采样数量

        Returns:
            其他用户的历史记录
        """
        if not self.all_users_history:
            return []

        other_users = [h for h in self.all_users_history if h != self.user_history]
        if not other_users:
            return []

        random_user = random.choice(other_users)
        return random_user[:n] if len(random_user) >= n else random_user

    def format_history_table(self, history: List[Dict], max_comment_len: int = 50) -> str:
        """
        将历史记录格式化为表格

        Args:
            history: 历史记录列表
            max_comment_len: 评论最大显示长度

        Returns:
            格式化的表格字符串
        """
        if not history:
            return "（无历史记录）"

        lines = ["| 电影名称 | 导演 | 类型 | 评分 | 评论 |",
                 "|----------|------|------|------|------|"]

        for item in history:
            name = item.get('movie_name', '未知')[:20]
            director = item.get('director', '未知')[:10]
            tags = item.get('tags', '')[:20]
            rating = item.get('rating', '?')
            comment = item.get('comment', '')[:max_comment_len]
            if len(item.get('comment', '')) > max_comment_len:
                comment += '...'

            lines.append(f"| {name} | {director} | {tags} | {rating} | {comment} |")

        return "\n".join(lines)

    def format_history_list(self, history: List[Dict], style: str = 'simple') -> str:
        """
        将历史记录格式化为列表

        Args:
            history: 历史记录列表
            style: 格式风格 - 'simple', 'detailed', 'compact'

        Returns:
            格式化的列表字符串
        """
        if not history:
            return "（无历史记录）"

        lines = []
        for i, item in enumerate(history, 1):
            name = item.get('movie_name', '未知')
            rating = item.get('rating', '?')
            comment = item.get('comment', '')

            if style == 'simple':
                lines.append(f"{i}. {name} - 评分: {rating}")
            elif style == 'detailed':
                director = item.get('director', '未知')
                tags = item.get('tags', '')
                lines.append(f"{i}. {name} (导演: {director}, 类型: {tags}) - 评分: {rating}")
                if comment:
                    lines.append(f"   评论: {comment[:100]}")
            elif style == 'compact':
                lines.append(f"{name}({rating}分)")

        return "\n".join(lines)

    def get_user_stats(self) -> Dict[str, Any]:
        """
        获取用户评分统计

        Returns:
            包含平均分、评分分布等统计信息的字典
        """
        if not self.user_history:
            return {'avg': 0, 'count': 0, 'min': 0, 'max': 0, 'distribution': {}}

        ratings = [item.get('rating', 0) for item in self.user_history]
        distribution = {}
        for r in ratings:
            distribution[r] = distribution.get(r, 0) + 1

        return {
            'avg': sum(ratings) / len(ratings),
            'count': len(ratings),
            'min': min(ratings),
            'max': max(ratings),
            'distribution': distribution
        }


# ============================================================
# Prompt执行引擎
# ============================================================

MAX_PROMPT_LENGTH = 1024 * 1024


def execute_prompt_function(code: str, ctx: PromptContext) -> tuple:
    """执行学生编写的Prompt生成函数"""
    safe_globals = {
        '__builtins__': {
            'len': len, 'range': range, 'min': min, 'max': max,
            'sum': sum, 'sorted': sorted, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'list': list,
            'dict': dict, 'set': set, 'tuple': tuple, 'int': int,
            'float': float, 'str': str, 'bool': bool, 'abs': abs,
            'round': round, 'print': print, 'random': random,
            're': re, 'json': json,
        }
    }
    safe_locals = {}
    exec(code, safe_globals, safe_locals)

    if 'generate_prompt' not in safe_locals:
        raise ValueError("代码中必须定义 generate_prompt(ctx) 函数")

    result = safe_locals['generate_prompt'](ctx)

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("generate_prompt 函数必须返回 (system_prompt, user_prompt) 元组")

    system_prompt, user_prompt = result
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise ValueError("system_prompt 和 user_prompt 必须是字符串")

    return system_prompt, user_prompt


def truncate_prompt(prompt: str, max_length: int = MAX_PROMPT_LENGTH) -> str:
    """截断Prompt到最大长度限制"""
    if len(prompt) <= max_length:
        return prompt
    return prompt[-max_length:]


def parse_llm_response(response: str) -> int:
    """解析LLM响应，提取评分"""
    pattern = r'\[Result:\s*([1-5])\s*\]'
    matches = re.findall(pattern, response)
    if matches:
        return int(matches[-1])

    pattern_loose = r'\[Result:\s*(\d+)\s*\]'
    matches_loose = re.findall(pattern_loose, response)
    if matches_loose:
        rating = int(matches_loose[-1])
        if 1 <= rating <= 5:
            return rating

    numbers = re.findall(r'[1-5]', response)
    if numbers:
        return int(numbers[-1])

    return 3


# ============================================================
# 评分计算函数
# ============================================================

def calculate_token_cost(input_tokens: int, output_tokens: int, pricing: dict) -> float:
    """计算token成本"""
    input_k = input_tokens / 1000
    output_k = output_tokens / 1000

    if input_k < 32:
        input_cost = input_k * pricing['input_tier1']
    else:
        input_cost = input_k * pricing['input_tier2']

    if output_k < 1:
        output_cost = output_k * pricing['output_tier1']
    else:
        output_cost = output_k * pricing['output_tier2']

    return input_cost + output_cost


def calculate_revenue(predictions: list, answers: list, revenue_table: dict) -> float:
    """计算总收益"""
    pred_dict = {(p['user_id'], p['movie_id']): p['rating'] for p in predictions}
    ans_dict = {(a['user_id'], a['movie_id']): a['rating'] for a in answers}

    total_revenue = 0
    for key in pred_dict:
        if key in ans_dict:
            error = abs(pred_dict[key] - ans_dict[key])
            if error < 0.5:
                total_revenue += revenue_table['error_lt_0.5']
            elif error <= 1.0:
                total_revenue += revenue_table['error_lte_1.0']
            else:
                total_revenue += revenue_table['error_gt_1.0']

    return total_revenue


def calculate_token_efficiency_score(total_input_tokens: int, total_output_tokens: int, num_samples: int = 1) -> float:
    """计算Token效率得分"""
    if num_samples <= 0:
        num_samples = 1

    total_tokens = total_input_tokens + total_output_tokens
    avg_tokens = total_tokens / num_samples

    if avg_tokens < 400:
        return 100.0

    score = max(0, 100 * (1 - math.log(avg_tokens / 400) / math.log(10)))
    return score


def calculate_profit_rate_score(total_revenue: float, total_cost: float) -> float:
    """计算收益率得分"""
    if total_cost <= 0:
        return 100.0

    profit_rate = (total_revenue - total_cost) / total_cost

    if profit_rate > 0:
        score = min(100, profit_rate * 6 + 60)
    elif profit_rate > -0.9:
        score = (profit_rate + 0.9) / 0.9 * 60
    else:
        score = 0

    return max(0, score)


def calculate_metrics(predictions: list, ground_truth: list) -> dict:
    """计算评测指标"""
    import numpy as np

    pred_dict = {(p['user_id'], p['movie_id']): float(p['rating']) for p in predictions}
    truth_dict = {(t['user_id'], t['movie_id']): float(t['rating']) for t in ground_truth}

    common_keys = set(pred_dict.keys()) & set(truth_dict.keys())
    if not common_keys:
        return None

    y_pred = np.array([pred_dict[k] for k in common_keys])
    y_true = np.array([truth_dict[k] for k in common_keys])
    errors = np.abs(y_pred - y_true)

    return {
        'num_samples': len(common_keys),
        'mae': float(np.mean(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'accuracy_exact': float(np.mean(y_pred == y_true) * 100),
        'accuracy_0.5': float(np.mean(errors <= 0.5) * 100),
        'accuracy_1.0': float(np.mean(errors <= 1.0) * 100),
    }


# ============================================================
# API调用
# ============================================================

def call_glm_api(system_prompt: str, user_prompt: str, api_key: str,
                 model: str = "glm-4.5-air", sample_index: int = 0, mock: bool = False) -> tuple:
    """
    调用GLM API

    Returns:
        tuple: (response_text, input_tokens, output_tokens)
    """
    est_input_tokens = len(system_prompt) + len(user_prompt)

    if mock:
        result = f"[Result:{random.randint(1, 5)}]"
        return result, est_input_tokens, 50

    try:
        import zhipuai

        if not api_key or api_key == 'your_api_key_here':
            print(f"  [警告] 未配置有效的API Key, 使用随机预测")
            result = f"[Result:{random.randint(1, 5)}]"
            return result, est_input_tokens, 50

        client = zhipuai.ZhipuAI(api_key=api_key)

        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}}
        )
        elapsed = time.time() - start_time

        result = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else est_input_tokens
        output_tokens = response.usage.completion_tokens if response.usage else len(result)

        print(f"  [样本{sample_index}] API响应成功, 耗时{elapsed:.1f}s, 输入{input_tokens}tok, 输出{output_tokens}tok")
        return result, input_tokens, output_tokens

    except Exception as e:
        print(f"  [样本{sample_index}] API调用失败: {e}")
        result = f"[Result:{random.randint(1, 5)}]"
        return result, est_input_tokens, 50


# ============================================================
# 主流程
# ============================================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='学生自测脚本 - 评测Prompt代码的各项得分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python self_test.py --data-dir . --code my_prompt.py --api-key xxx
  python self_test.py --data-dir . --code my_prompt.py --api-key xxx --type eval
  python self_test.py --data-dir . --code my_prompt.py --api-key xxx --limit 5
  python self_test.py --data-dir . --code my_prompt.py --mock
        """
    )
    parser.add_argument('--data-dir', type=str, default='.', help='数据集目录路径, 默认当前目录')
    parser.add_argument('--code', type=str, required=True, help='Prompt代码文件路径(必须包含generate_prompt函数)')
    parser.add_argument('--api-key', type=str, default='', help='智谱AI API Key')
    parser.add_argument('--type', type=str, default='test', choices=['test', 'eval'], help='评测类型: test或eval')
    parser.add_argument('--limit', type=int, default=0, help='只评测前N个样本(0表示全部)')
    parser.add_argument('--mock', action='store_true', help='使用随机预测代替API调用(调试用)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    code_path = Path(args.code)

    # 验证文件存在
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)
    if not code_path.exists():
        print(f"错误: 代码文件不存在: {code_path}")
        sys.exit(1)

    # 加载代码
    print("=" * 60)
    print("学生自测脚本 - LLM电影评分预测")
    print("=" * 60)

    print(f"\n[1/6] 加载Prompt代码: {code_path}")
    prompt_code = code_path.read_text(encoding='utf-8')

    if 'def generate_prompt' not in prompt_code:
        print("错误: 代码中必须定义 generate_prompt(ctx) 函数")
        sys.exit(1)
    print(f"  代码长度: {len(prompt_code)} 字符")

    # 加载数据
    print(f"\n[2/6] 加载数据集 ({data_dir})")
    train_data = load_json(data_dir / 'train.json')
    movies_info = load_json(data_dir / 'movies_info.json')
    all_users_history = [user['history'] for user in train_data.get('users', [])]

    if args.type == 'test':
        test_input = load_json(data_dir / 'test_input.json')
        test_answer = load_json(data_dir / 'test_answer.json')
        print(f"  评测类型: 测试集 (test)")
    else:
        test_input = load_json(data_dir / 'eval_input.json')
        test_answer = load_json(data_dir / 'eval_answer.json')
        print(f"  评测类型: 评分集 (eval)")

    if args.limit > 0:
        test_input = test_input[:args.limit]
        test_answer = test_answer[:args.limit]
        print(f"  限制样本数: {args.limit}")

    print(f"  训练用户数: {len(all_users_history)}")
    print(f"  待评测样本数: {len(test_input)}")
    print(f"  电影信息数: {len(movies_info)}")

    # 配置
    pricing = {
        'input_tier1': 0.5,
        'input_tier2': 1.0,
        'output_tier1': 2.0,
        'output_tier2': 4.0,
    }
    revenue_table = {
        'error_lt_0.5': 10.0,
        'error_lte_1.0': 2.0,
        'error_gt_1.0': 0.1,
    }
    fixed_cost_per_call = 1.0
    weights = {
        'mae': 0.10,
        'rmse': 0.10,
        'accuracy_exact': 0.10,
        'accuracy_1.0': 0.30,
        'token_efficiency': 0.10,
        'profit_rate': 0.50,
    }

    # 逐样本预测
    print(f"\n[3/6] 开始预测 ({'Mock模式' if args.mock else 'API模式'})")
    predictions = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_token_cost = 0.0
    success_count = 0
    error_count = 0

    for i, item in enumerate(test_input):
        user_id = item['user_id']
        context_history = item.get('context_history', [])
        target_movie = item.get('target_movie', {})
        movie_id = target_movie.get('movie_id', '')

        ctx = PromptContext(
            user_history=context_history,
            target_movie=target_movie,
            movies_info=movies_info,
            all_users_history=all_users_history
        )

        try:
            system_prompt, user_prompt = execute_prompt_function(prompt_code, ctx)

            if len(system_prompt) > MAX_PROMPT_LENGTH:
                system_prompt = truncate_prompt(system_prompt)
            if len(user_prompt) > MAX_PROMPT_LENGTH:
                user_prompt = truncate_prompt(user_prompt)

            response, input_tokens, output_tokens = call_glm_api(
                system_prompt, user_prompt, args.api_key,
                model="glm-4.5-air", sample_index=i, mock=args.mock
            )

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            token_cost = calculate_token_cost(input_tokens, output_tokens, pricing)
            total_token_cost += token_cost

            rating = parse_llm_response(response)

            predictions.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating
            })
            success_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(test_input):
                print(f"  进度: {i + 1}/{len(test_input)}")

        except Exception as e:
            error_count += 1
            print(f"  [错误] 样本{i}({target_movie.get('movie_name', '?')}): {e}")
            predictions.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': 3
            })

    print(f"  完成: 成功{success_count}, 失败{error_count}")

    # 计算指标
    print(f"\n[4/6] 计算评测指标")
    metrics = calculate_metrics(predictions, test_answer)

    if metrics is None:
        print("错误: 没有共同样本, 无法计算指标")
        sys.exit(1)

    # 计算各项得分
    print(f"\n[5/6] 计算各项得分")
    total_fixed_cost = success_count * fixed_cost_per_call
    total_cost = total_token_cost + total_fixed_cost
    total_revenue = calculate_revenue(predictions, test_answer, revenue_table)

    mae_score = max(0, (1 - metrics['mae'] / 4)) * 100
    rmse_score = max(0, (1 - metrics['rmse'] / 4)) * 100
    acc_exact_score = metrics['accuracy_exact']
    acc_10_score = metrics['accuracy_1.0']
    token_eff_score = calculate_token_efficiency_score(total_input_tokens, total_output_tokens, len(test_input))
    profit_rate_score = calculate_profit_rate_score(total_revenue, total_cost)

    raw_score = (
        weights['mae'] * mae_score +
        weights['rmse'] * rmse_score +
        weights['accuracy_exact'] * acc_exact_score +
        weights['accuracy_1.0'] * acc_10_score +
        weights['token_efficiency'] * token_eff_score +
        weights['profit_rate'] * profit_rate_score
    )
    final_score = min(100, raw_score)

    profit_rate = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0

    # 打印结果
    print(f"\n[6/6] 评测结果")
    print("=" * 60)
    print("评测报告")
    print("=" * 60)

    print(f"\n样本数量: {metrics['num_samples']}")
    print(f"评测类型: {'测试集' if args.type == 'test' else '评分集'}")

    print(f"\n【核心指标】")
    print(f"  MAE  (平均绝对误差):  {metrics['mae']:.4f}")
    print(f"  RMSE (均方根误差):    {metrics['rmse']:.4f}")

    print(f"\n【准确率】")
    print(f"  完全准确率:           {metrics['accuracy_exact']:.2f}%")
    print(f"  误差<=0.5 准确率:     {metrics['accuracy_0.5']:.2f}%")
    print(f"  误差<=1.0 准确率:     {metrics['accuracy_1.0']:.2f}%")

    print(f"\n【Token统计】")
    print(f"  总输入Token:          {total_input_tokens}")
    print(f"  总输出Token:          {total_output_tokens}")
    print(f"  平均每次输入Token:    {total_input_tokens // max(success_count, 1)}")
    print(f"  平均每次输出Token:    {total_output_tokens // max(success_count, 1)}")

    print(f"\n【成本与收益】")
    print(f"  Token成本:            {total_token_cost:.4f} 元")
    print(f"  固定支出:             {total_fixed_cost:.2f} 元")
    print(f"  总成本:               {total_cost:.4f} 元")
    print(f"  总收益:               {total_revenue:.2f} 元")
    print(f"  收益率:               {profit_rate:.4f}")

    print(f"\n【各项得分】")
    print(f"  MAE得分       (权重10%):  {mae_score:.2f}")
    print(f"  RMSE得分      (权重10%):  {rmse_score:.2f}")
    print(f"  精准命中率    (权重10%):  {acc_exact_score:.2f}")
    print(f"  准确率<=1.0   (权重30%):  {acc_10_score:.2f}")
    print(f"  Token效率     (权重10%):  {token_eff_score:.2f}")
    print(f"  收益率        (权重50%):  {profit_rate_score:.2f}")

    print(f"\n{'=' * 60}")
    print(f"  原始得分:               {raw_score:.2f}")
    print(f"  综合得分(截断100):      {final_score:.2f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
