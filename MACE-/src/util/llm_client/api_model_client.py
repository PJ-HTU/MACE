import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient

class APIModelClient(BaseLLMClient):
    """Client for interacting with LLMs through API endpoints.
    
    Supports multiple model types: Claude, OpenAI, DeepSeek, and OpenRouter.
    """
    
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)
        
        self.config = config
        self.url = config["url"]
        model = config["model"]
        stream = config.get("stream", False)
        top_p = config.get("top-p", 0.7)
        temperature = config.get("temperature", 0.95)
        max_tokens = config.get("max_tokens", 3200)
        seed = config.get("seed", None)
        api_key = config["api_key"]
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)
        
        self.model_type = self._identify_model_type(self.url)
        self._setup_headers_and_payload(model, stream, max_tokens, temperature, top_p, seed, api_key)

    def _identify_model_type(self, url: str) -> str:
        """Identify model type based on URL."""
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
            return 'openai'

    def _setup_headers_and_payload(self, model, stream, max_tokens, temperature, top_p, seed, api_key):
        """Configure headers and payload based on model type."""
        
        if self.model_type == 'claude':
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
            if seed is not None:
                self.payload["seed"] = seed
                
        elif self.model_type == 'deepseek':
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
            if seed is not None:
                self.payload["seed"] = seed
                
        elif self.model_type == 'openrouter':
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
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
            
            if seed is not None:
                self.payload["seed"] = seed

    def _parse_response(self, response_text: str) -> str:
        """Parse API response based on model type."""
        response_data = json.loads(response_text)

        if self.model_type == 'claude':
            return response_data["content"][0]["text"]
            
        elif self.model_type in ['openai', 'deepseek', 'openrouter']:
            return response_data["choices"][-1]["message"]["content"]

    def reset(self, output_dir: str=None) -> None:
        """Reset conversation state and optionally update output directory."""
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        """Send current conversation to API and return model response."""
        
        formatted_messages = []
        for msg in self.messages:
            text_content = msg["content"][0]["text"] if msg["content"] else ""
            formatted_messages.append({
                "role": msg["role"],
                "content": text_content
            })
        self.payload["messages"] = formatted_messages
        
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        response_content = self._parse_response(response.text)
        
        return response_content