import os
import sys
import subprocess
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 定义 4 组实验配置
experiments = [
    # Group A: 基准组 (GPT-4o + 英文提示词)
    {"name": "Group_A_GPT4o_English", "model_type": "gpt-4o", "prompt_dir": "inference/prompts/en"},

    # Group B: 中文 GPT 组 (GPT-4o + 中文提示词)
    {"name": "Group_B_GPT4o_Chinese", "model_type": "gpt-4o", "prompt_dir": "inference/prompts/cn"},

    # Group C: 英文 Qwen 组 (Qwen-Max + 英文提示词)
    {"name": "Group_C_QwenMax_English", "model_type": "qwen3-max", "prompt_dir": "inference/prompts/en"},

    # Group D: 中文 Qwen 组 (Qwen-Max + 中文提示词) -> 重点关注
    {"name": "Group_D_QwenMax_Chinese", "model_type": "qwen3-max", "prompt_dir": "inference/prompts/cn"},
]


def run_task(exp_config):
    print(f"\n{'=' * 60}")
    print(f"🚀 正在启动实验: {exp_config['name']}")
    print(f"   🤖 模型类型: {exp_config['model_type']}")
    print(f"   📂 提示词目录: {exp_config['prompt_dir']}")
    print(f"{'=' * 60}")

    # 1. 设置当前子进程的环境变量
    # 这些变量会被 config.py 读取，从而覆盖默认设置
    env = os.environ.copy()
    env["LLM_MODEL_TYPE"] = exp_config["model_type"]
    env["PROMPT_DIR"] = exp_config["prompt_dir"]

    try:
        # 2. 启动 main.py 子进程
        # 使用 sys.executable 确保用的是当前的 python 环境 (conda env)
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,  # 允许输入
            stdout=subprocess.PIPE,  # 捕获输出
            stderr=subprocess.PIPE,  # 捕获错误
            text=True,  # 以文本模式处理
            encoding='utf-8',
            errors='replace',
            env=env,  # 注入修改后的环境变量
            cwd=os.path.dirname(os.path.abspath(__file__))  # 确保在当前目录运行
        )

        # 3. 模拟用户输入: 发送 "2" (选择波动率因子) 并回车
        print("   ...正在自动选择 '2: 波动率因子' 并运行挖掘流程...")
        stdout, stderr = process.communicate(input="2\n")

        # 4. 打印部分日志 (可选)
        # print(stdout) # 如果不想刷屏，可以注释掉这行

        if stderr:
            print(f"❌ 警告/错误信息:\n{stderr}")

        # 5. 重命名结果文件，防止被覆盖
        target_result_file = "result.txt"
        if os.path.exists(target_result_file):
            # 保存为 result_Group_A_....txt
            new_name = f"result_{exp_config['name']}.txt"
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(target_result_file, new_name)
            print(f"✅ 实验成功！结果已保存至: {new_name}")
        else:
            print("⚠️ 未找到 result.txt，该组实验可能未生成有效结果。")

    except Exception as e:
        print(f"💥 实验运行异常: {e}")


if __name__ == "__main__":
    print("开始执行 2x2 对照组实验...")
    for exp in experiments:
        run_task(exp)
        print("⏳ 冷却 5 秒，准备下一组...\n")
        time.sleep(5)
    print("\n🎉 所有实验已完成！请查看生成的 4 个 result_*.txt 文件。")
