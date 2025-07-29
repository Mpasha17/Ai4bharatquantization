"""
FastAPI Backend Service for Quantized Airavata Model
Provides REST API endpoints for text generation with performance metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import logging
import time
import uvicorn
from model_quantizer import AiravataQuantizer
import torch
import psutil
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None

# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(default=512, ge=50, le=2048, description="Maximum generation length")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    quantization: Optional[str] = Field(default=None, description="Quantization type (fp16, int8, int4)")

class GenerationResponse(BaseModel):
    generated_text: str
    latency: float
    tokens_per_second: float
    memory_usage: Dict[str, Any]
    quantization_type: str

class ModelLoadRequest(BaseModel):
    quantization_type: str = Field(default="int8", description="Quantization type: fp16, int8, or int4")
    device_map: str = Field(default="auto", description="Device mapping strategy")

class BenchmarkRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of test prompts")
    num_runs: int = Field(default=5, ge=1, le=20, description="Number of benchmark runs per prompt")
    quantization_types: List[str] = Field(default=["fp16", "int8", "int4"], description="Quantization types to test")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    quantization_type: Optional[str]
    system_info: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Airavata FastAPI server...")
    yield
    # Shutdown
    global model_instance
    if model_instance:
        del model_instance.model
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Airavata FastAPI server shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Airavata Quantized Model API",
    description="FastAPI service for quantized Airavata 7B model with performance metrics",
    version="1.0.0",
    lifespan=lifespan
)

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "cuda_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.device_count() > 0 else None
        })
    
    return info

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Airavata Quantized Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model_instance
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None and model_instance.model is not None,
        quantization_type=getattr(model_instance, 'quantization_config', None),
        system_info=get_system_info()
    )

@app.post("/load_model")
async def load_model(request: ModelLoadRequest):
    """Load model with specified quantization"""
    global model_instance
    
    try:
        # Clean up existing model
        if model_instance:
            del model_instance.model
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Initialize new quantizer
        model_instance = AiravataQuantizer()
        
        # Load tokenizer
        if not model_instance.load_tokenizer():
            raise HTTPException(status_code=500, detail="Failed to load tokenizer")
        
        # Load model with quantization
        if not model_instance.load_model(
            quantization_type=request.quantization_type,
            device_map=request.device_map
        ):
            raise HTTPException(status_code=500, detail=f"Failed to load model with {request.quantization_type} quantization")
        
        memory_usage = model_instance.get_model_memory_usage()
        
        return {
            "status": "success",
            "message": f"Model loaded with {request.quantization_type} quantization",
            "quantization_type": request.quantization_type,
            "memory_usage": memory_usage
        }
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the loaded model"""
    global model_instance
    
    if model_instance is None or model_instance.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first using /load_model")
    
    try:
        # If quantization is specified and different from current, reload model
        if request.quantization and hasattr(model_instance, 'quantization_config'):
            current_quant = getattr(model_instance, 'quantization_config', None)
            if current_quant != request.quantization:
                load_req = ModelLoadRequest(quantization_type=request.quantization)
                await load_model(load_req)
        
        # Measure generation time
        start_time = time.time()
        generated_text = model_instance.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Estimate tokens per second (rough approximation)
        estimated_tokens = len(generated_text.split()) * 1.3
        tokens_per_second = estimated_tokens / latency if latency > 0 else 0
        
        # Get memory usage
        memory_usage = model_instance.get_model_memory_usage()
        
        return GenerationResponse(
            generated_text=generated_text,
            latency=latency,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage,
            quantization_type=getattr(model_instance, 'quantization_config', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/benchmark")
async def benchmark_model(request: BenchmarkRequest):
    """Benchmark model performance across different quantization types"""
    results = {}
    
    for quantization_type in request.quantization_types:
        try:
            logger.info(f"Benchmarking {quantization_type} quantization...")
            
            # Load model with current quantization
            load_req = ModelLoadRequest(quantization_type=quantization_type)
            await load_model(load_req)
            
            # Run benchmark
            if model_instance:
                benchmark_results = model_instance.benchmark_inference(
                    test_prompts=request.prompts,
                    num_runs=request.num_runs
                )
                results[quantization_type] = benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark failed for {quantization_type}: {e}")
            results[quantization_type] = {"error": str(e)}
    
    return {
        "benchmark_results": results,
        "test_prompts": request.prompts,
        "num_runs": request.num_runs
    }

@app.get("/model_info")
async def get_model_info():
    """Get information about the currently loaded model"""
    global model_instance
    
    if model_instance is None:
        return {"status": "no_model_loaded"}
    
    info = {
        "model_name": model_instance.model_name,
        "model_loaded": model_instance.model is not None,
        "tokenizer_loaded": model_instance.tokenizer is not None,
        "quantization_config": str(model_instance.quantization_config) if model_instance.quantization_config else None,
        "memory_usage": model_instance.get_model_memory_usage() if model_instance.model else None
    }
    
    return info

@app.post("/unload_model")
async def unload_model():
    """Unload the current model to free memory"""
    global model_instance
    
    if model_instance:
        del model_instance.model
        del model_instance
        model_instance = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"status": "success", "message": "Model unloaded successfully"}
    else:
        return {"status": "info", "message": "No model was loaded"}

# Example usage endpoints
@app.get("/examples")
async def get_examples():
    """Get example requests for testing the API"""
    return {
        "load_model_example": {
            "quantization_type": "int8",
            "device_map": "auto"
        },
        "generate_example": {
            "prompt": "What is artificial intelligence?",
            "max_length": 256,
            "temperature": 0.7
        },
        "benchmark_example": {
            "prompts": [
                "Explain machine learning in simple terms.",
                "What are the benefits of renewable energy?",
                "Write a short poem about technology."
            ],
            "num_runs": 3,
            "quantization_types": ["int8", "int4"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )