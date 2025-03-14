# ACAS: Advanced Contract Analysis System
Advanced Contract Analysis System: A comprehensive legal contract analysis system using generative AI. The project implements various NLP techniques, prompt engineering approaches (CoT, TroT, GoT), Retrieval-Augmented Generation (RAG), multimodal inputs, QLoRA fine-tuning, and evaluation frameworks.

## System Requirements

- CUDA 11.7 or higher (used 12.6)
- Python 3.9 or higher (used 3.10)
- 16GB+ system RAM
- 8GB+ GPU VRAM
- 50GB+ free disk space

## Setup and Installation

Models used:
```bash
Mistral-7B-Instruct-v0.3.Q5_K_M.gguf
Meta-Llama-3-8B-Instruct.Q5_K_M.gguf
```

Dataset used:
https://www.atticusprojectai.org/cuad

Create a dedicated virtual environment for the project:

```bash
# Create and activate a conda environment (recommended)
conda create -n legal-ai python=3.9
conda activate legal-ai

# Install core dependencies
pip install transformers
pip install torchtorchvision torchaudio
pip install bitsandbytes
pip install peft
pip install sentence-transformers
pip install llama-cpp-python
pip install faiss-gpu
pip install pandas numpy matplotlib seaborn tqdm
pip install fastapi uvicorn python-multipart
pip install spacy nltk datasets rouge_score
pip install scikit-learn
pip install SpeechRecognition pytesseract pdf2image
pip install networkx

# Install SpaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

Additional system dependencies:

```bash
# For OCR
sudo apt-get install tesseract-ocr libtesseract-dev
# For PDF processing
sudo apt-get install poppler-utils
```

## Project Structure

The project is implemented in several phases:

- NLP analysis, prompt engineering techniques, and RAG implementation for legal contracts
- Multimodal agents capable of processing text, voice, images, and PDFs
- Fine-tuning language models with QLoRA for legal domain adaptation
- Comprehensive evaluation framework for model assessment

## Running the System

### Preprocessing and Preparing the System

Run the python files:

```bash
python contract_data_extraction.py
python reasoning_frameworks.py
python rag_system.py
```

### Multimodal Agent

Start the backend server:

```bash
python multimodal_agent.py
```

Access the frontend interface:

```bash
python -m http.server 8080
# Go to http://localhost:8080/multimodal-agent-frontend.html
```

### Fine-Tuning with QLoRA

Run the fine-tuning process:

```bash
python qlora-fine-tuning.py \
    --base_model ./llama3-8b-hf \
    --dataset_path ./processed_contracts.pkl \
    --output_dir ./legal-contract-model \
    --cuad_path ./CUAD_v1/CUAD_v1.json
```

Test the fine-tuned model:

```bash
python qlora-fine-tuning.py \
    --base_model ./llama3-8b-hf \
    --dataset_path ./processed_contracts.pkl \
    --output_dir ./legal-contract-model \
    --cuad_path ./CUAD_v1/CUAD_v1.json \
    --test
```

### Evaluation Framework

Create evaluation datasets:

```bash
python evaluation-framework.py \
    --create_dataset \
    --cuad_path ./CUAD_v1/CUAD_v1.json \
    --dataset_output ./legal_evaluation_data.json
```

Run evaluation:

```bash
python evaluation-framework.py \
    --model_path ./legal-contract-model.gguf \
    --reference_model ./Meta-Llama-3-8B-Instruct.Q5_K_M.gguf \
    --evaluation_data ./legal_evaluation_data.json \
    --output_dir ./evaluation_results \
    --use_gpu
```

### Integration Script

To run the entire system:

```bash
# Run the multimodal agent only
python integration.py --mode agent

# Run fine-tuning only
python integration.py --mode finetune

# Run evaluation only
python integration.py --mode evaluate

# Run all components in sequence
python integration.py --mode all
```

## Troubleshooting

- **CUDA Out of Memory Errors**: Reduce batch size, use gradient accumulation, or try using 4-bit quantization
- **OCR Quality Issues**: Improve image preprocessing, use higher resolution scans
- **Model Performance Issues**: Increase training epochs, add more domain-specific examples
- **Speech Recognition Accuracy**: Reduce background noise, speak clearly
- **Integration Issues**: Ensure all components use the same model format and paths
