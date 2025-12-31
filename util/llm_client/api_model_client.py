# 导入必要的模块：os用于文件操作，json用于JSON数据处理，requests用于发送HTTP请求，以及从基础类模块导入BaseLLMClient基类
import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient

# 定义APIModelClient类，继承自BaseLLMClient，用于通过API接口与大语言模型交互
class APIModelClient(BaseLLMClient):
    # 初始化方法：接收配置字典、提示词目录和输出目录，调用父类初始化方法并设置API相关参数
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)  # 调用父类的初始化方法
        
        # 保存完整配置以供后续使用
        self.config = config

        self.url = config["url"]  # 从配置中获取API请求地址
        model = config["model"]  # 从配置中获取模型名称
        stream = config.get("stream", False)  # 从配置中获取流式输出标志，默认为False
        top_p = config.get("top-p", 0.7)  # 从配置中获取top-p参数，默认为0.7
        temperature = config.get("temperature", 0.95)  # 从配置中获取温度参数，默认为0.95
        max_tokens = config.get("max_tokens", 3200)  # 从配置中获取最大token数，默认为3200
        seed = config.get("seed", None)  # 从配置中获取随机种子，默认为None
        api_key = config["api_key"]  # 从配置中获取API密钥
        self.max_attempts = config.get("max_attempts", 50)  # 从配置中获取最大尝试次数，默认为50
        self.sleep_time = config.get("sleep_time", 60)  # 从配置中获取重试间隔时间，默认为60秒
        
        # 根据URL识别模型类型
        self.model_type = self._identify_model_type(self.url)
        
        # 根据模型类型设置不同的headers和payload
        self._setup_headers_and_payload(model, stream, max_tokens, temperature, top_p, seed, api_key)

    def _identify_model_type(self, url: str) -> str:
        """根据URL识别模型类型"""
        url_lower = url.lower()
        
        if 'anthropic' in url_lower or 'claude' in url_lower:
            return 'claude'
        elif 'openai' in url_lower or 'chatgpt' in url_lower:
            return 'openai'
        elif 'deepseek' in url_lower:
            return 'deepseek'
        elif 'openrouter' in url_lower:
            return 'openrouter'
        else:
            # 默认使用OpenAI格式（因为很多模型都兼容OpenAI格式）
            return 'openai'

    def _setup_headers_and_payload(self, model, stream, max_tokens, temperature, top_p, seed, api_key):
        """根据模型类型设置headers和payload"""
        
        if self.model_type == 'claude':
            # Claude (Anthropic) API 配置
            self.headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            self.payload = {
                "model": model,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
        elif self.model_type == 'openai':
            # OpenAI/ChatGPT API 配置
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            self.payload = {
                "model": model,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            # 如果指定了seed，添加到payload中
            if seed is not None:
                self.payload["seed"] = seed
                
        elif self.model_type == 'deepseek':
            # DeepSeek API 配置
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            self.payload = {
                "model": model,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            # DeepSeek可能有自己特殊的参数
            if seed is not None:
                self.payload["seed"] = seed
                
        elif self.model_type == 'openrouter':
            # OpenRouter API 配置
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # OpenRouter可选的额外头部信息（从config中获取）
            http_referer = self.config.get("http_referer")
            x_title = self.config.get("x_title")
            
            if http_referer:
                self.headers["HTTP-Referer"] = http_referer
            if x_title:
                self.headers["X-Title"] = x_title
                
            self.payload = {
                "model": model,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # 如果指定了seed，添加到payload中
            if seed is not None:
                self.payload["seed"] = seed

    def _parse_response(self, response_text: str) -> str:
        """根据模型类型解析响应"""
        response_data = json.loads(response_text)

        if self.model_type == 'claude':
            # Claude响应格式
            return response_data["content"][0]["text"]
            
        elif self.model_type in ['openai', 'deepseek', 'openrouter']:
            # OpenAI、DeepSeek和OpenRouter响应格式
            return response_data["choices"][-1]["message"]["content"]

    # 重置对话状态：清空消息列表，若指定新输出目录则更新并创建目录（重写父类方法）
    def reset(self, output_dir: str=None) -> None:
        self.messages = []  # 清空存储对话历史的列表
        if output_dir is not None:
            self.output_dir = output_dir  # 更新输出目录
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在，不存在则创建

    # 单次聊天方法（重写父类抽象方法）：发送当前对话历史到API并返回模型响应
    def chat_once(self) -> str:
        
        # 转换消息格式：将 content 从 [{"type": "text", "text": "..."}] 提取为纯字符串
        formatted_messages = []
        for msg in self.messages:
            # 假设消息内容中只有文本类型，取第一个文本内容
            text_content = msg["content"][0]["text"] if msg["content"] else ""
            formatted_messages.append({
                "role": msg["role"],
                "content": text_content  # 仅保留字符串内容
            })
        self.payload["messages"] = formatted_messages  # 使用转换后的消息
        
        # 发送POST请求到API

        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
 
        # 根据模型类型解析响应
        response_content = self._parse_response(response.text)
        
        return response_content  # 返回模型响应的文本内容