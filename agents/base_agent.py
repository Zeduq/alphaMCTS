"""
from abc import ABC, abstractmethod
import openai
import json
# 修正: 需要从 'typing' 模块导入 Any 和 Dict
from typing import Any, Dict
from config import OPENAI_API_KEY, LLM_MODEL

class BaseAgent(ABC):
    # 所有LLM Agent的抽象基类。
    def __init__(self, prompt_path: str):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        # 执行Agent的任务并返回解析后的结果。
        pass

    def _call_llm(self, formatted_prompt: str) -> Dict[str, Any]:

        # 调用LLM并解析JSON输出的辅助方法。

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7, # 在论文的上下文中，温度可以根据任务调整
            )
            # 修正: 正确访问响应内容的方式是 response.choices[0].message.content
            content = response.choices[0].message.content
            if not content:
                print("Error: LLM returned empty content.")
                return None
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Received content: {content}")
            return None
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in _call_llm: {e}")
            return None
"""
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai

# from config import LLM_MODEL, GROQ_API_KEY, BASE_URL
from config import LLM_MODEL, OPENAI_API_KEY, BASE_URL


class BaseAgent(ABC):
    """所有LLM Agent的抽象基类。"""
    def __init__(self, prompt_path: str):
        self.client = openai.OpenAI(
            #api_key=GROQ_API_KEY,
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
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

    def _call_llm(self, formatted_prompt: str) -> Dict[str, Any]:
        """
        调用LLM并解析JSON输出的辅助方法。
        """
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
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Received content: {content}")
            return None
        except openai.APIError as e:
            print(f"API Error from provider: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in _call_llm: {e}")
            return None
