"""
Local Models Manager - Ollama integration for DeepSeek, Qwen, etc.
"""
import os
import subprocess
import requests
import json
import logging
from typing import List, Dict

class LocalModelsManager:
    def __init__(self, ollama_host=None):
        self.ollama_host = ollama_host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.logger = logging.getLogger(__name__)
        
        # Recommended models for financial analysis
        self.recommended_models = {
            "reasoning": "deepseek-r1-distill-llama-8b",
            "structured_data": "qwen2.5:14b", 
            "embeddings": "bge-m3",
            "fallback": "llama3.1:8b"
        }
    
    def check_ollama_status(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_installed_models(self) -> List[str]:
        """Get list of installed models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def install_model(self, model_name: str):
        """Install a model via Ollama"""
        try:
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {model_name}")
                return True
            else:
                self.logger.error(f"Failed to install {model_name}: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error installing {model_name}: {e}")
            return False
    
    def setup_recommended_models(self):
        """Setup all recommended models for annual reports"""
        installed = self.list_installed_models()
        
        for purpose, model in self.recommended_models.items():
            if model not in installed:
                print(f"Installing {model} for {purpose}...")
                self.install_model(model)
            else:
                print(f"✅ {model} already installed ({purpose})")
    
    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for specific task"""
        task_mapping = {
            "financial_analysis": "reasoning",
            "data_extraction": "structured_data", 
            "embeddings": "embeddings",
            "general": "fallback"
        }
        
        model_type = task_mapping.get(task, "fallback")
        return self.recommended_models[model_type]

if __name__ == "__main__":
    manager = LocalModelsManager()
    print("Checking Ollama status...")
    
    if manager.check_ollama_status():
        print("✅ Ollama is running")
        print("Installed models:", manager.list_installed_models())
    else:
        print("❌ Ollama is not running. Please start Ollama first.")
