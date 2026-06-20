#!/usr/bin/env python3
"""
Quick verification script to test if everything works
Run this to ensure your system is ready for the presentation
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 AGRI-RAG System Verification\n")
print("="*50)

checks_passed = 0
checks_total = 0

# Check 1: Environment variables
print("\n✓ Check 1: Environment Variables")
checks_total += 1
required_env = ["GROQ_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
for env_var in required_env:
    if os.getenv(env_var):
        print(f"  ✓ {env_var} configured")
        checks_passed += 1
    else:
        print(f"  ✗ {env_var} missing - add to .env")

# Check 2: Ground truth dataset
print("\n✓ Check 2: Ground Truth Dataset")
checks_total += 1
gt_file = Path("data/ragas_ground_truth.json")
if gt_file.exists():
    import json
    with open(gt_file) as f:
        gt_data = json.load(f)
    num_questions = len(gt_data.get('questions', []))
    print(f"  ✓ Ground truth file exists with {num_questions} questions")
    checks_passed += 1
else:
    print(f"  ✗ Ground truth file not found at {gt_file}")

# Check 3: Required modules
print("\n✓ Check 3: Python Dependencies")
checks_total += 1
try:
    import streamlit
    import torch
    import transformers
    from langchain_groq import ChatGroq
    import faiss
    print("  ✓ All required modules installed")
    checks_passed += 1
except ImportError as e:
    print(f"  ✗ Missing module: {e}")

# Check 4: Evaluation module
print("\n✓ Check 4: Evaluation Module")
checks_total += 1
try:
    from src.evaluation.ragas_evaluator import RAGASEvaluator, HallucinationDetector
    print("  ✓ Evaluation modules loaded")
    checks_passed += 1
except Exception as e:
    print(f"  ✗ Error loading evaluation module: {e}")

# Check 5: Web UI
print("\n✓ Check 5: Web UI")
checks_total += 1
ui_file = Path("ui/demo_comparison.py")
if ui_file.exists():
    print(f"  ✓ Web UI file exists")
    checks_passed += 1
else:
    print(f"  ✗ Web UI not found at {ui_file}")

# Check 6: Data directories
print("\n✓ Check 6: Data Directories")
checks_total += 1
data_dirs = ["data/faiss_db", "data/chroma_db"]
all_exist = True
for d in data_dirs:
    path = Path(d)
    if path.exists():
        print(f"  ✓ {d} exists")
    else:
        print(f"  ! {d} does not exist (will be created on first ingestion)")
        all_exist = False
if all_exist:
    checks_passed += 1
else:
    print("  ! Data directories will be created when needed")
    checks_passed += 0.5

# Summary
print("\n" + "="*50)
print(f"\n✓ Verification Complete: {int(checks_passed)}/{checks_total} checks passed")

if checks_passed >= checks_total - 1:
    print("\n🎉 System is ready! Run:")
    print("   streamlit run ui/demo_comparison.py --server.port 8510")
elif checks_passed >= checks_total - 2:
    print("\n⚠️  Minor issues detected. You may need to:")
    print("   1. Set up GROQ_API_KEY in .env")
    print("   2. Ensure Neo4j is running")
else:
    print("\n❌ Please fix the issues above before running")
