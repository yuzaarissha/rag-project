"""
Configuration management for RAG system with dynamic model selection.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import ollama


@dataclass
class ModelConfig:
    """Configuration for model selection."""
    llm_model: str = ""  # Will be auto-detected from available models
    embedding_model: str = ""  # Will be auto-detected from available models
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**data)


class ConfigManager:
    """Manages configuration persistence and model discovery."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> ModelConfig:
        """Load configuration from file or create default with auto-detection."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                config = ModelConfig.from_dict(data)
                # Validate that models are still available
                if not self.is_model_available(config.llm_model) or not self.is_model_available(config.embedding_model):
                    print("Saved models not available, auto-detecting...")
                    return self._create_default_config()
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> ModelConfig:
        """Create default configuration with auto-detected models."""
        available_models = self.get_available_models()
        
        # Auto-select first available LLM model
        llm_model = available_models['llm'][0] if available_models['llm'] else ""
        
        # Auto-select first available embedding model
        embedding_model = available_models['embedding'][0] if available_models['embedding'] else ""
        
        config = ModelConfig(llm_model=llm_model, embedding_model=embedding_model)
        
        # Save the auto-detected config
        if llm_model and embedding_model:
            self.config = config
            self.save_config()
            print(f"Auto-detected models - LLM: {llm_model}, Embedding: {embedding_model}")
        else:
            print("❌ Could not auto-detect models:")
            if not llm_model:
                print("  - No LLM models found. Install one with: ollama pull <llm_model>")
            if not embedding_model:
                print("  - No embedding models found. Install one with: ollama pull <embedding_model>")
        
        return config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models from Ollama, categorized by type."""
        try:
            models = ollama.list()
            model_names = []
            
            # Handle both dict and object response formats
            if hasattr(models, 'models'):
                # New format: object with models attribute
                for model in models.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            elif isinstance(models, dict) and 'models' in models:
                # Old format: dict with models key
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
            
            # Categorize models (basic heuristic)
            llm_models = []
            embedding_models = []
            
            for model_name in model_names:
                if any(embed_keyword in model_name.lower() for embed_keyword in ['embed', 'embedding', 'nomic', 'mxbai']):
                    embedding_models.append(model_name)
                else:
                    llm_models.append(model_name)
            
            return {
                'llm': llm_models,
                'embedding': embedding_models
            }
        except Exception as e:
            print(f"Error getting available models: {e}")
            return {'llm': [], 'embedding': []}
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        try:
            models = ollama.list()
            model_names = []
            
            # Handle both dict and object response formats
            if hasattr(models, 'models'):
                # New format: object with models attribute
                for model in models.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            elif isinstance(models, dict) and 'models' in models:
                # Old format: dict with models key
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
            
            return model_name in model_names
        except Exception:
            return False
    
    def update_llm_model(self, model_name: str) -> bool:
        """Update LLM model selection."""
        if self.is_model_available(model_name):
            self.config.llm_model = model_name
            self.save_config()
            return True
        return False
    
    def update_embedding_model(self, model_name: str) -> bool:
        """Update embedding model selection."""
        if self.is_model_available(model_name):
            self.config.embedding_model = model_name
            self.save_config()
            return True
        return False
    
    def get_current_config(self) -> ModelConfig:
        """Get current configuration."""
        return self.config
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        try:
            models = ollama.list()
            
            # Handle both dict and object response formats
            if hasattr(models, 'models'):
                # New format: object with models attribute
                for model in models.models:
                    if hasattr(model, 'model') and model.model == model_name:
                        return {
                            'name': model.model,
                            'size': model.size if hasattr(model, 'size') else 'Unknown',
                            'modified_at': model.modified_at.isoformat() if hasattr(model, 'modified_at') else 'Unknown',
                            'digest': model.digest if hasattr(model, 'digest') else 'Unknown'
                        }
            elif isinstance(models, dict) and 'models' in models:
                # Old format: dict with models key
                for model in models['models']:
                    if isinstance(model, dict) and model.get('name') == model_name:
                        return model
            
            return None
        except Exception:
            return None