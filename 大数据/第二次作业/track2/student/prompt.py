
def generate_prompt(ctx):
    """
    生成Prompt的主函数
    
    Args:
        ctx: PromptContext对象，提供数据和工具方法
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    # 获取用户最近的5条历史记录
    recent_history = ctx.get_history_sample(5, 'recent')
    
    # 格式化历史表格
    history_table = ctx.format_history_table(recent_history)
    
    # 获取目标电影信息
    movie = ctx.target_movie
    
    # 构建System Prompt
    system_prompt = """你是一个电影评分预测专家。根据用户的历史评分记录，预测用户对新电影的评分。

评分标准：
- 1分：非常差
- 2分：较差
- 3分：一般
- 4分：较好
- 5分：非常好

请在最后一行输出 [Result:X]，其中X是你的预测评分（1-5的整数）。"""
    
    # 构建User Prompt
    user_prompt = f"""## 用户历史评分记录

{history_table}

## 待预测电影信息

- 电影名称：{movie.get('name', '未知')}
- 导演：{movie.get('director', '未知')}
- 类型：{movie.get('tags', '未知')}
- 简介：{movie.get('summary', '无简介')[:500]}

## 任务要求

请预测该用户对此电影的评分（1-5分，整数）。

请在最后一行输出 [Result:X]，其中X是你的预测评分。

[Result:"""
    
    return system_prompt, user_prompt
