"""
API Testing Script for Airavata FastAPI Server
Tests all endpoints and demonstrates usage
"""

import requests
import json
import time
from typing import Dict, Any

class APITester:
    """Test the FastAPI server endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print("Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Health check passed: {result['status']}")
            return result
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return {"error": str(e)}
    
    def test_load_model(self, quantization_type: str = "int8") -> Dict[str, Any]:
        """Test model loading"""
        print(f"Testing model loading with {quantization_type}...")
        try:
            payload = {
                "quantization_type": quantization_type,
                "device_map": "auto"
            }
            response = self.session.post(f"{self.base_url}/load_model", json=payload)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Model loaded successfully with {quantization_type}")
            return result
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return {"error": str(e)}
    
    def test_text_generation(self, prompt: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Test text generation"""
        print("Testing text generation...")
        try:
            payload = {
                "prompt": prompt,
                "max_length": 256,
                "temperature": 0.7
            }
            response = self.session.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Text generation successful")
            print(f"  Generated: {result['generated_text'][:100]}...")
            print(f"  Latency: {result['latency']:.2f}s")
            print(f"  Throughput: {result['tokens_per_second']:.2f} tokens/s")
            return result
        except Exception as e:
            print(f"✗ Text generation failed: {e}")
            return {"error": str(e)}
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        print("Testing model info...")
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Model info retrieved")
            return result
        except Exception as e:
            print(f"✗ Model info failed: {e}")
            return {"error": str(e)}
    
    def test_benchmark(self, quantization_types: list = ["int8"]) -> Dict[str, Any]:
        """Test benchmarking endpoint"""
        print("Testing benchmark...")
        try:
            payload = {
                "prompts": [
                    "Explain machine learning in simple terms.",
                    "What are the benefits of renewable energy?",
                    "How does photosynthesis work?"
                ],
                "num_runs": 3,
                "quantization_types": quantization_types
            }
            response = self.session.post(f"{self.base_url}/benchmark", json=payload)
            response.raise_for_status()
            result = response.json()
            print(f"✓ Benchmark completed")
            return result
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            return {"error": str(e)}
    
    def test_examples(self) -> Dict[str, Any]:
        """Test examples endpoint"""
        print("Testing examples...")
        try:
            response = self.session.get(f"{self.base_url}/examples")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Examples retrieved")
            return result
        except Exception as e:
            print(f"✗ Examples failed: {e}")
            return {"error": str(e)}
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        print("=" * 60)
        print("AIRAVATA API TESTING SUITE")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Health check
        results["health"] = self.test_health_check()
        print()
        
        # Test 2: Examples
        results["examples"] = self.test_examples()
        print()
        
        # Test 3: Load model with INT8
        results["load_int8"] = self.test_load_model("int8")
        print()
        
        # Test 4: Model info
        results["model_info"] = self.test_model_info()
        print()
        
        # Test 5: Text generation
        results["generation"] = self.test_text_generation("Explain the concept of neural networks.")
        print()
        
        # Test 6: Multiple generations with different prompts
        test_prompts = [
            "What is quantum computing?",
            "How does blockchain work?",
            "Explain renewable energy sources."
        ]
        
        results["multiple_generations"] = []
        for i, prompt in enumerate(test_prompts):
            print(f"Testing generation {i+1}/3...")
            gen_result = self.test_text_generation(prompt)
            results["multiple_generations"].append(gen_result)
            time.sleep(1)  # Small delay between requests
        print()
        
        # Test 7: Load model with INT4 (if supported)
        print("Testing INT4 quantization...")
        results["load_int4"] = self.test_load_model("int4")
        if "error" not in results["load_int4"]:
            results["generation_int4"] = self.test_text_generation("What is machine learning?")
        print()
        
        # Test 8: Benchmark (limited to avoid long runtime)
        results["benchmark"] = self.test_benchmark(["int8"])
        print()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = 0
        
        for test_name, result in results.items():
            if test_name == "multiple_generations":
                for i, gen_result in enumerate(result):
                    total += 1
                    if "error" not in gen_result:
                        passed += 1
                        print(f"✓ Multiple generation {i+1}: PASSED")
                    else:
                        print(f"✗ Multiple generation {i+1}: FAILED")
            else:
                total += 1
                if "error" not in result:
                    passed += 1
                    print(f"✓ {test_name}: PASSED")
                else:
                    print(f"✗ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Save detailed results
        with open("api_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: api_test_results.json")
        
        return results

def main():
    """Main function"""
    print("Starting API tests...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("You can start it with: python fastapi_server.py")
    print()
    
    # Wait for user confirmation
    input("Press Enter to continue with testing...")
    
    tester = APITester()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/")
        print("✓ Server is running")
    except:
        print("✗ Server is not running. Please start the server first.")
        return
    
    # Run tests
    tester.run_full_test_suite()

if __name__ == "__main__":
    main()