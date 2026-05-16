"""
Download NLI Model (DeBERTa-v3) for Agri-RAG
Run this to download the NLI model for fact pruning.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("Downloading NLI Model (DeBERTa-v3)")
print("="*60)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "cross-encoder/nli-deberta-v3-base"
    print(f"Model: {model_name}")
    print("Downloading... (this may take 2-5 minutes)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print("NLI model downloaded successfully!")
    print(f"\nModel saved to: ~/.cache/huggingface/")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTry installing with: pip install transformers torch")

print("\nNLI Model ready!")