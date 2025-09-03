# agents/base_agent.py

from abc import ABC, abstractmethod
import openai
import json
import re
from typing import Any, Dict, Optional

# 直接从我们的配置文件中导入配置
from config import LLM_MODEL, OPENAI_API_KEY


class BaseAgent(ABC):
    """所有LLM Agent的抽象基类。"""

    def __init__(self, prompt_path: str):
        # 检查我们从config.py导入的API密钥是否有效
        if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
            raise ValueError("请在 config.py 文件中设置您的有效 OPENAI_API_KEY。")

        # 修正：在初始化客户端时，显式地将我们导入的密钥传递给 api_key 参数
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file not found at {prompt_path}")
            raise

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行Agent的任务并返回解析后的结果。"""
        pass

    def _call_llm(self, formatted_prompt: str) -> Optional[Dict[str, Any]]:
        """
        调用LLM并解析JSON输出的辅助方法。
        """
        content = ""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if not content:
                print("Error: LLM returned empty content.")
                return None

            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                clean_content = match.group(1)
            else:
                clean_content = content.strip()

            return json.loads(clean_content)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Received content: {content}")
            return None
        except openai.APIError as e:
            # 这个except块会捕获到401错误并打印出来
            print(f"API Error from provider: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in _call_llm: {e}")
            return None