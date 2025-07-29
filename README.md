# Airavata Model Quantization & FastAPI Backend

Easily quantize and serve the Ai4bharat/Airavata 7B model with a FastAPI backend.

---

## Key Features
- Quantize the Airavata model (FP16, INT8, INT4)
- FastAPI server for easy API access
- Performance metrics and benchmarking

---

## Quick Start for Local only if you have gpu

### 1. Clone & Install
```bash
# Clone the repo
git clone https://github.com/Mpasha17/Ai4bharatquantization
cd quant

# (Recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
```bash
python fastapi_server.py
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

### 3. Load a Quantized Model
**Note:** INT8/INT4 quantization requires Linux + NVIDIA GPU (CUDA). On Mac/CPU, only FP16 or full precision may work.

```bash
# Example: Load INT4 quantized model (Linux + GPU only)
curl -X POST "http://localhost:8000/load_model" \
     -H "Content-Type: application/json" \
     -d '{"quantization_type": "int4", "device_map": "auto"}'

# Example: Load FP16 model (CPU/Mac)
curl -X POST "http://localhost:8000/load_model" \
     -H "Content-Type: application/json" \
     -d '{"quantization_type": "fp16", "device_map": "cpu"}'
```

### 4. Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is AI?", "max_length": 128}'
```

### 5. Run Performance Tests
```bash
# Run comprehensive benchmarks across all quantization methods
python performance_tester.py
```

Alternatively, you can test performance via API:
```bash
# Benchmark specific quantization methods
curl -X POST "http://localhost:8000/benchmark" \
     -H "Content-Type: application/json" \
     -d '{
           "prompts": ["What is AI?", "Explain quantum computing"],
           "num_runs": 5,
           "quantization_types": ["fp16", "int8", "int4"]
         }'
```

---

## Running on AWS (Recommended for Quantized Models)

1. **Launch an EC2 Instance**
   - **Recommended:** Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023), g4dn.xlarge or better (NVIDIA GPU, 16GB+ RAM)
   - Add at least 100GB disk space

2. **Connect & Setup**
   ```bash
   ssh -i <your-key.pem> ubuntu@<your-ec2-public-ip>
   sudo apt update && sudo apt install -y python3-venv git
   git clone https://github.com/Mpasha17/Ai4bharatquantization
   cd quant
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Check CUDA: python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Start the Server**
   ```bash
   python fastapi_server.py
   # Open port 8000 in AWS Security Group to access the API
   ```

4. **Load the Model (INT4)**
   ```bash
   curl -X POST "http://<ec2-public-ip>:8000/load_model" \
        -H "Content-Type: application/json" \
        -d '{"quantization_type": "int4", "device_map": "auto"}'
   ```

5. **Generate Text**
   ```bash
   curl -X POST "http://<ec2-public-ip>:8000/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Your question here"}'
   ```

6. **Run Performance Tests**
   ```bash
   # Run comprehensive benchmarks
   python performance_tester.py
   
   # Or via API
   curl -X POST "http://<ec2-public-ip>:8000/benchmark" \
        -H "Content-Type: application/json" \
        -d '{
              "prompts": ["What is AI?", "Explain quantum computing"],
              "num_runs": 5,
              "quantization_types": ["fp16", "int8", "int4"]
            }'
   ```

---

## Notes & Limitations
- **INT8/INT4 quantization only works on Linux with NVIDIA GPU (CUDA).**
- **Mac/CPU users:** Only FP16 or full precision is supported, and performance will be slow.
- For best results, use a cloud GPU instance (AWS, GCP, Azure) with at least 16GB VRAM.
- The API is documented at `/docs` once the server is running.

---

## File Overview
- `fastapi_server.py` — FastAPI backend
- `model_quantizer.py` — Model quantization logic
- `performance_tester.py` — Benchmarking
- `test_api.py` — API tests

