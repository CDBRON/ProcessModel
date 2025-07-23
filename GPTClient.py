import asyncio
import os
import time
import logging
from google import genai
from google.genai import types
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
class Config:
    Gemini_API_KEY: str = os.getenv("Gemini_API_KEY")
    Gemini_MODEL: str = "gemini-2.0-flash"
    TEMPERATURE: float = 0.1

class GPTClient:
    def __init__(self, api_key: str = Config.Gemini_API_KEY, model: str = Config.Gemini_MODEL,
                 temperature: float = Config.TEMPERATURE):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = genai.Client(api_key=self.api_key)
        #self.a_client = self.client  # 使用普通Client实例，确保异步方法调用

    def chat_completion(self, combined_prompt, temperature: float = None) :
        temp = temperature if temperature is not None else self.temperature
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[combined_prompt],
                    config=types.GenerateContentConfig(temperature=temp)
                )
                return response.text
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and "429" in str(e):
                    retry_after = 31  # 默认等待时间
                    # 尝试解析RetryInfo中的等待时间
                    error_dict = e.args[0].to_dict() if hasattr(e, 'to_dict') else {}
                    if 'details' in error_dict.get('error', {}):
                        for detail in error_dict['error']['details']:
                            if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                retry_after = int(float(detail['retryDelay'].split('s')[0]))
                    logger.warning(f"API请求被限流，将在{retry_after}秒后重试({retry_count+1}/{max_retries})")
                    time.sleep(retry_after)
                    retry_count +=1
                else:
                    raise
        raise RuntimeError(f"达到最大重试次数({max_retries})，API调用失败：{str(e)}")
