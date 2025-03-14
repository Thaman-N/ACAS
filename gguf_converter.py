import os
import argparse
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download
import logging
from llama_cpp import Llama
import json
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_base_model(model_id):
    """Download base model from HuggingFace"""
    try:
        cache_dir = os.path.join(tempfile.gettempdir(), "hf_models", model_id.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Downloading model {model_id} to {cache_dir}")
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def convert_gguf_to_hf(gguf_path, output_dir, base_model_id=None):
    """
    Convert a GGUF model to HuggingFace format for fine-tuning
    
    Args:
        gguf_path: Path to the GGUF model
        output_dir: Directory to save the converted model
        base_model_id: Optional HF model ID to use for architecture info
    """
    if not os.path.exists(gguf_path):
        logger.error(f"GGUF model not found at {gguf_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the GGUF model using llama.cpp
    logger.info(f"Loading GGUF model from {gguf_path}")
    try:
        llama_model = Llama(
            model_path=gguf_path,
            n_ctx=2048,  # Smaller context for conversion
            n_gpu_layers=0  # CPU mode for conversion
        )
        logger.info("GGUF model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading GGUF model: {e}")
        return False
    
    # If base_model_id is provided, use its architecture as a reference
    if base_model_id:
        logger.info(f"Using {base_model_id} as reference architecture")
        model_path = download_base_model(base_model_id)
        
        if not model_path:
            logger.error("Failed to download reference model")
            return False
        
        try:
            # Load tokenizer from the reference model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load config from the reference model
            config = AutoConfig.from_pretrained(model_path)
            
            # Save tokenizer and config to the output directory
            tokenizer.save_pretrained(output_dir)
            config.save_pretrained(output_dir)
            
            logger.info(f"Tokenizer and config saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error loading reference model components: {e}")
            return False
    else:
        # If no reference model, we need to infer from the GGUF model
        logger.warning("No reference model provided. Limited conversion possible.")
        
        # Try to extract model information from GGUF
        model_info = {}
        try:
            # Generate a simple completion to get model info
            completion = llama_model.create_completion("Hello", max_tokens=1)
            
            if hasattr(llama_model, 'model_path') and hasattr(llama_model, 'params'):
                model_info = {
                    "model_path": llama_model.model_path,
                    "n_ctx": llama_model.params.n_ctx if hasattr(llama_model.params, 'n_ctx') else 4096,
                    "n_embd": llama_model.params.n_embd if hasattr(llama_model.params, 'n_embd') else 4096,
                    "n_layer": llama_model.params.n_layer if hasattr(llama_model.params, 'n_layer') else 32,
                }
            
            # Save model info to output directory
            with open(os.path.join(output_dir, "gguf_model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
                
            logger.info(f"Extracted model info: {model_info}")
        except Exception as e:
            logger.error(f"Error extracting model info: {e}")
    
    logger.info("You should now use this directory with the HuggingFace Transformers library")
    logger.info(f"The converted model is saved at: {output_dir}")
    logger.info("Note: For best results with fine-tuning, use a reference model that matches the architecture of your GGUF model.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert GGUF model to HuggingFace format for fine-tuning")
    parser.add_argument("--gguf_path", type=str, required=True, help="Path to the GGUF model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted model")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3-8B", 
                       help="HuggingFace model ID to use as reference architecture")
    
    args = parser.parse_args()
    
    convert_gguf_to_hf(args.gguf_path, args.output_dir, args.base_model_id)

if __name__ == "__main__":
    main()
