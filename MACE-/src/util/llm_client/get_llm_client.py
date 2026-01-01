import json
from src.util.llm_client.base_llm_client import BaseLLMClient

def get_llm_client(
        config_file: str,
        prompt_dir: str=None,
        output_dir: str=None,
        ) -> BaseLLMClient:
    """Create and return an LLM client instance based on configuration file.
    
    Args:
        config_file: Path to the configuration file
        prompt_dir: Directory containing prompt templates
        output_dir: Directory for output files
        
    Returns:
        BaseLLMClient: LLM client instance
    """
    config = json.load(open(config_file))
    llm_type = config["type"]
    
    if llm_type == "azure_apt":
        from src.util.llm_client.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "api_model":
        from src.util.llm_client.api_model_client import APIModelClient
        llm_client = APIModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "local_model":
        from src.util.llm_client.local_model_client import LocalModelClient
        llm_client = LocalModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    
    return llm_client