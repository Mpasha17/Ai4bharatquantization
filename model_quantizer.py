"""
Model Quantization Utilities for Airavata Model
Supports INT8, INT4, and GGUF quantization methods
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import logging
from typing import Optional, Dict, Any
import psutil
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AiravataQuantizer:
    """Handles quantization of Airavata model with different precision levels"""
    
    def __init__(self, model_name: str = "ai4bharat/Airavata"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.quantization_config = None
        
    def load_tokenizer(self):
        """Load the tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded successfully for {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False
    
    def create_quantization_config(self, quantization_type: str = "int8") -> BitsAndBytesConfig:
        """Create quantization configuration"""
        if quantization_type == "int8":
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quantization_type == "int4":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        self.quantization_config = config
        return config
    
    def load_model(self, quantization_type: str = "int8", device_map: str = "auto"):
        """Load model with specified quantization"""
        try:
            # Create quantization config
            if quantization_type != "fp16":
                quant_config = self.create_quantization_config(quantization_type)
            else:
                quant_config = None
            
            # Load model with CPU offloading support
            if quantization_type == "fp16":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True,
                    offload_folder="offload"
                )
            else:
                # Enable CPU offloading for quantized models
                if quantization_type == "int8":
                    quant_config.llm_int8_enable_fp32_cpu_offload = True
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    offload_folder="offload"
                )
            
            logger.info(f"Model loaded successfully with {quantization_type} quantization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
            }
        
        # System memory
        process = psutil.Process()
        system_memory = {
            "system_memory_gb": process.memory_info().rss / 1024**3,
            "system_memory_percent": process.memory_percent()
        }
        
        return {**gpu_memory, **system_memory}
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            return "Error: Model or tokenizer not loaded"
        
        try:
            # Format prompt according to Airavata's chat format
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move to same device as model
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error during generation: {str(e)}"
    
    def benchmark_inference(self, test_prompts: list, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark inference performance"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        results = {
            "latencies": [],
            "throughput_tokens_per_second": [],
            "memory_usage": self.get_model_memory_usage()
        }
        
        for prompt in test_prompts:
            prompt_latencies = []
            prompt_throughputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                response = self.generate_text(prompt, max_length=256)
                end_time = time.time()
                
                latency = end_time - start_time
                # Estimate tokens (rough approximation)
                tokens_generated = len(response.split()) * 1.3  # rough token estimate
                throughput = tokens_generated / latency if latency > 0 else 0
                
                prompt_latencies.append(latency)
                prompt_throughputs.append(throughput)
            
            results["latencies"].extend(prompt_latencies)
            results["throughput_tokens_per_second"].extend(prompt_throughputs)
        
        # Calculate averages
        results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
        results["avg_throughput"] = sum(results["throughput_tokens_per_second"]) / len(results["throughput_tokens_per_second"])
        
        return results

def test_quantization_methods():
    """Test different quantization methods and compare performance"""
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot."
    ]
    
    quantization_methods = ["fp16", "int8", "int4"]
    results = {}
    
    for method in quantization_methods:
        print(f"\n=== Testing {method.upper()} Quantization ===")
        
        quantizer = AiravataQuantizer()
        
        # Load tokenizer
        if not quantizer.load_tokenizer():
            print(f"Failed to load tokenizer for {method}")
            continue
        
        # Load model with quantization
        if not quantizer.load_model(quantization_type=method):
            print(f"Failed to load model with {method} quantization")
            continue
        
        # Benchmark
        benchmark_results = quantizer.benchmark_inference(test_prompts, num_runs=3)
        results[method] = benchmark_results
        
        print(f"Average Latency: {benchmark_results.get('avg_latency', 'N/A'):.2f}s")
        print(f"Average Throughput: {benchmark_results.get('avg_throughput', 'N/A'):.2f} tokens/s")
        print(f"Memory Usage: {benchmark_results.get('memory_usage', {})}")
        
        # Clean up
        del quantizer.model
        del quantizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

if __name__ == "__main__":
    # Run quantization tests
    results = test_quantization_methods()
    print("\n=== Quantization Comparison Results ===")
    for method, result in results.items():
        print(f"{method.upper()}: Latency={result.get('avg_latency', 'N/A'):.2f}s, "
              f"Throughput={result.get('avg_throughput', 'N/A'):.2f} tokens/s")