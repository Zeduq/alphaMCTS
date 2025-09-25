import json
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai

from config import LLM_MODEL, OPENAI_API_KEY, BASE_URL


class BaseAgent(ABC):
    """所有LLM Agent的抽象基类。"""
    def __init__(self, prompt_path: str):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
            # 设置60秒的超时时间，以应对网络延迟或服务器繁忙
            timeout=60.0,
            # 设置2次自动重试，以处理瞬时网络问题
            max_retries=2,
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

            # 检查并移除Markdown代码块标记
            if content.strip().startswith("```json"):
                content = content.strip()[7:-3].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            original_content_for_debug = response.choices[0].message.content
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"Received content: {original_content_for_debug}")
            return None
        except openai.APIError as e:
            print(f"API Error from provider: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in _call_llm: {e}")
            return None
