"""
Performance Testing and Benchmarking Suite for Quantized Airavata Model
Comprehensive testing of latency, throughput, and memory usage across quantization methods
"""

import time
import json
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import torch
import psutil
import logging
from model_quantizer import AiravataQuantizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.test_prompts = [
            "What is artificial intelligence and how does it work?",
            "Explain the concept of machine learning in simple terms.",
            "What are the main benefits of renewable energy sources?",
            "Describe the process of photosynthesis in plants.",
            "How do neural networks learn from data?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain quantum computing and its potential applications.",
            "What are the ethical considerations in AI development?",
            "How does blockchain technology work?",
            "What is the impact of climate change on global ecosystems?"
        ]
        
        self.quantization_methods = ["fp16", "int8", "int4"]
        self.results = {}
    
    def measure_model_loading_time(self, quantization_type: str) -> Dict[str, Any]:
        """Measure model loading time and memory usage"""
        logger.info(f"Measuring loading time for {quantization_type}")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        quantizer = AiravataQuantizer()
        
        # Measure tokenizer loading
        start_time = time.time()
        tokenizer_success = quantizer.load_tokenizer()
        tokenizer_time = time.time() - start_time
        
        if not tokenizer_success:
            return {"error": "Failed to load tokenizer"}
        
        # Measure model loading
        start_time = time.time()
        model_success = quantizer.load_model(quantization_type=quantization_type)
        model_loading_time = time.time() - start_time
        
        if not model_success:
            return {"error": f"Failed to load model with {quantization_type}"}
        
        # Get memory usage after loading
        memory_usage = quantizer.get_model_memory_usage()
        
        # Clean up
        del quantizer.model
        del quantizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "tokenizer_loading_time": tokenizer_time,
            "model_loading_time": model_loading_time,
            "total_loading_time": tokenizer_time + model_loading_time,
            "memory_usage": memory_usage
        }
    
    def measure_inference_performance(self, quantization_type: str, num_runs: int = 10) -> Dict[str, Any]:
        """Measure detailed inference performance"""
        logger.info(f"Measuring inference performance for {quantization_type}")
        
        quantizer = AiravataQuantizer()
        
        # Load model
        if not quantizer.load_tokenizer() or not quantizer.load_model(quantization_type=quantization_type):
            return {"error": f"Failed to load model with {quantization_type}"}
        
        results = {
            "latencies": [],
            "throughputs": [],
            "memory_snapshots": [],
            "prompt_results": {}
        }
        
        # Test each prompt multiple times
        for i, prompt in enumerate(self.test_prompts[:5]):  # Use first 5 prompts for detailed testing
            prompt_latencies = []
            prompt_throughputs = []
            
            for run in range(num_runs):
                # Measure memory before generation
                memory_before = quantizer.get_model_memory_usage()
                
                # Generate text
                start_time = time.time()
                generated_text = quantizer.generate_text(prompt, max_length=256, temperature=0.7)
                end_time = time.time()
                
                # Measure memory after generation
                memory_after = quantizer.get_model_memory_usage()
                
                latency = end_time - start_time
                
                # Estimate throughput (tokens per second)
                estimated_tokens = len(generated_text.split()) * 1.3  # rough approximation
                throughput = estimated_tokens / latency if latency > 0 else 0
                
                prompt_latencies.append(latency)
                prompt_throughputs.append(throughput)
                
                results["latencies"].append(latency)
                results["throughputs"].append(throughput)
                results["memory_snapshots"].append({
                    "before": memory_before,
                    "after": memory_after
                })
            
            results["prompt_results"][f"prompt_{i}"] = {
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "avg_latency": statistics.mean(prompt_latencies),
                "std_latency": statistics.stdev(prompt_latencies) if len(prompt_latencies) > 1 else 0,
                "avg_throughput": statistics.mean(prompt_throughputs),
                "std_throughput": statistics.stdev(prompt_throughputs) if len(prompt_throughputs) > 1 else 0
            }
        
        # Calculate overall statistics
        results["overall_stats"] = {
            "avg_latency": statistics.mean(results["latencies"]),
            "median_latency": statistics.median(results["latencies"]),
            "std_latency": statistics.stdev(results["latencies"]) if len(results["latencies"]) > 1 else 0,
            "min_latency": min(results["latencies"]),
            "max_latency": max(results["latencies"]),
            "avg_throughput": statistics.mean(results["throughputs"]),
            "median_throughput": statistics.median(results["throughputs"]),
            "std_throughput": statistics.stdev(results["throughputs"]) if len(results["throughputs"]) > 1 else 0,
            "min_throughput": min(results["throughputs"]),
            "max_throughput": max(results["throughputs"])
        }
        
        # Clean up
        del quantizer.model
        del quantizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def run_comprehensive_benchmark(self, num_runs: int = 10) -> Dict[str, Any]:
        """Run comprehensive benchmark across all quantization methods"""
        logger.info("Starting comprehensive benchmark...")
        
        benchmark_results = {
            "system_info": self.get_system_info(),
            "test_config": {
                "num_runs": num_runs,
                "num_test_prompts": len(self.test_prompts[:5]),
                "quantization_methods": self.quantization_methods
            },
            "results": {}
        }
        
        for quantization_type in self.quantization_methods:
            logger.info(f"Benchmarking {quantization_type}...")
            
            # Measure loading performance
            loading_results = self.measure_model_loading_time(quantization_type)
            
            # Measure inference performance
            inference_results = self.measure_inference_performance(quantization_type, num_runs)
            
            benchmark_results["results"][quantization_type] = {
                "loading": loading_results,
                "inference": inference_results
            }
            
            # Log summary
            if "error" not in loading_results and "error" not in inference_results:
                logger.info(f"{quantization_type} - Loading: {loading_results['total_loading_time']:.2f}s, "
                           f"Avg Latency: {inference_results['overall_stats']['avg_latency']:.2f}s, "
                           f"Avg Throughput: {inference_results['overall_stats']['avg_throughput']:.2f} tokens/s")
        
        return benchmark_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            })
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                info[f"cuda_device_{i}"] = {
                    "name": device_props.name,
                    "total_memory_gb": device_props.total_memory / 1024**3,
                    "multi_processor_count": device_props.multi_processor_count,
                    "major": device_props.major,
                    "minor": device_props.minor
                }
        
        return info
    
    def generate_performance_report(self, results: Dict[str, Any], output_file: str = "performance_report.json"):
        """Generate detailed performance report"""
        
        # Save raw results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("AIRAVATA MODEL QUANTIZATION PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # System info
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 20 + "\n")
            sys_info = results["system_info"]
            f.write(f"CPU Cores: {sys_info['cpu_count']}\n")
            f.write(f"Total Memory: {sys_info['memory_total_gb']:.2f} GB\n")
            f.write(f"CUDA Available: {sys_info['cuda_available']}\n")
            if sys_info['cuda_available']:
                f.write(f"CUDA Devices: {sys_info['cuda_device_count']}\n")
                for i in range(sys_info['cuda_device_count']):
                    device_key = f"cuda_device_{i}"
                    if device_key in sys_info:
                        device = sys_info[device_key]
                        f.write(f"  Device {i}: {device['name']} ({device['total_memory_gb']:.2f} GB)\n")
            f.write("\n")
            
            # Performance comparison
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Method':<8} {'Load Time':<12} {'Avg Latency':<14} {'Avg Throughput':<16} {'Memory (GB)':<12}\n")
            f.write("-" * 70 + "\n")
            
            for method, result in results["results"].items():
                if "error" not in result["loading"] and "error" not in result["inference"]:
                    load_time = result["loading"]["total_loading_time"]
                    avg_latency = result["inference"]["overall_stats"]["avg_latency"]
                    avg_throughput = result["inference"]["overall_stats"]["avg_throughput"]
                    
                    # Get memory usage (GPU if available, otherwise system)
                    memory_usage = result["loading"]["memory_usage"]
                    if "gpu_allocated_gb" in memory_usage:
                        memory = memory_usage["gpu_allocated_gb"]
                    else:
                        memory = memory_usage.get("system_memory_gb", 0)
                    
                    f.write(f"{method:<8} {load_time:<12.2f} {avg_latency:<14.2f} {avg_throughput:<16.2f} {memory:<12.2f}\n")
                else:
                    f.write(f"{method:<8} {'ERROR':<12} {'ERROR':<14} {'ERROR':<16} {'ERROR':<12}\n")
            
            f.write("\n")
            
            # Detailed statistics
            f.write("DETAILED STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for method, result in results["results"].items():
                if "error" not in result["inference"]:
                    f.write(f"\n{method.upper()} Quantization:\n")
                    stats = result["inference"]["overall_stats"]
                    f.write(f"  Latency - Mean: {stats['avg_latency']:.3f}s, Std: {stats['std_latency']:.3f}s\n")
                    f.write(f"  Latency - Min: {stats['min_latency']:.3f}s, Max: {stats['max_latency']:.3f}s\n")
                    f.write(f"  Throughput - Mean: {stats['avg_throughput']:.2f} tokens/s, Std: {stats['std_throughput']:.2f}\n")
                    f.write(f"  Throughput - Min: {stats['min_throughput']:.2f} tokens/s, Max: {stats['max_throughput']:.2f}\n")
        
        logger.info(f"Performance report saved to {output_file} and {summary_file}")
        return output_file, summary_file

def main():
    """Main function to run performance tests"""
    tester = PerformanceTester()
    
    print("Starting Airavata Model Quantization Performance Testing...")
    print("This may take several minutes depending on your hardware.\n")
    
    # Run comprehensive benchmark
    results = tester.run_comprehensive_benchmark(num_runs=5)
    
    # Generate reports
    json_file, summary_file = tester.generate_performance_report(results)
    
    print(f"\nPerformance testing completed!")
    print(f"Results saved to: {json_file}")
    print(f"Summary report: {summary_file}")
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print("-" * 40)
    for method, result in results["results"].items():
        if "error" not in result.get("loading", {}) and "error" not in result.get("inference", {}):
            load_time = result["loading"]["total_loading_time"]
            avg_latency = result["inference"]["overall_stats"]["avg_latency"]
            avg_throughput = result["inference"]["overall_stats"]["avg_throughput"]
            print(f"{method.upper()}: Load={load_time:.2f}s, Latency={avg_latency:.2f}s, Throughput={avg_throughput:.2f} tokens/s")
        else:
            print(f"{method.upper()}: ERROR during testing")

if __name__ == "__main__":
    main()