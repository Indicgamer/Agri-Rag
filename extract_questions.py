"""
Extract questions from triplet data for RAGAS evaluation
"""

import json
from pathlib import Path

# Read triplets
triplets_file = Path("data/exports/triplets.jsonl")
triplets = []

with open(triplets_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            triplets.append(json.loads(line.strip()))
        except:
            pass

print(f"Total triplets: {len(triplets)}")

# Get unique subjects from triplets
subjects = set()
relations = set()
objects = set()

for t in triplets:
    subjects.add(t.get('subject', '').lower())
    relations.add(t.get('relation', '').lower())
    objects.add(t.get('object', '').lower())

print(f"Unique subjects: {len(subjects)}")

# Define crop keywords from PDF
crops = ['rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'groundnut', 'sunflower', 
        'sorghum', 'cumbu', 'ragi', 'paddy', 'pulse', 'fodder', 'greengram', 
        'blackgram', 'redgram', 'bengalgram', 'soybean', 'castor', 'sesame',
        'chilli', 'tomato', 'onion', 'brinjal', 'mango', 'banana']

# Map relations to question patterns
relation_patterns = {
    'recommended': 'What is recommended for {subject}?',
    'variety': 'What are the varieties of {subject}?',
    'dose': 'What is the dose of {object} for {subject}?',
    'spacing': 'What is the spacing for {subject}?',
    'requires': 'What are the requirements for {subject}?',
    'related_to': 'What is related to {subject}?',
    'affects': 'What affects {subject}?',
    'management': 'How to manage {subject}?',
    'control': 'How to control {object} in {subject}?',
    'treatment': 'How to treat {object} in {subject}?',
    'provides': 'What does {subject} provide?',
    'causes': 'What causes {object} in {subject}?',
    'occurs_in': 'Where does {subject} occur?',
    'season': 'When is the season for {subject}?',
    'apply': 'How to apply {object} to {subject}?',
    'water': 'How much water does {subject} need?',
    'rainfall': 'What is the rainfall requirement for {subject}?',
    'temperature': 'What is the temperature requirement for {subject}?',
    'yield': 'What is the yield of {subject}?',
    'fertilizer': 'What fertilizer for {subject}?',
}

# Generate questions
generated_questions = []
unique_triplets = set()

for t in triplets:
    sub = t.get('subject', '').strip()
    rel = t.get('relation', '').lower()
    obj = t.get('object', '').strip()
    
    key = f"{sub}|{rel}|{obj}"
    if key in unique_triplets:
        continue
    unique_triplets.add(key)
    
    # Skip generic ones
    if not sub or not rel or not obj:
        continue
    if len(sub) < 3 or len(obj) < 3:
        continue
    
    # Check if crop-related
    is_crop = any(crop in sub.lower() for crop in crops)
    
    # Generate question based on relation
    rel_clean = rel.lower().replace('_', ' ')
    
    if 'recommend' in rel_clean:
        q = f"What is recommended for {sub}?"
    elif 'variet' in rel_clean:
        q = f"What are the varieties of {sub}?"
    elif 'dose' in rel_clean:
        q = f"What is the dose of {obj} for {sub}?"
    elif 'spacing' in rel_clean:
        q = f"What is the spacing for {sub}?"
    elif 'requir' in rel_clean:
        q = f"What are the requirements for {sub}?"
    elif 'water' in rel_clean:
        q = f"How much water does {sub} need?"
    elif 'rainfall' in rel_clean or 'rainfall' in rel_clean:
        q = f"What is the rainfall requirement for {sub}?"
    elif 'temperatur' in rel_clean or 'climate' in rel_clean:
        q = f"What is the temperature requirement for {sub}?"
    elif 'yield' in rel_clean:
        q = f"What is the yield of {sub}?"
    elif 'manag' in rel_clean:
        q = f"How to manage {sub}?"
    elif 'control' in rel_clean or 'treat' in rel_clean:
        q = f"How to control {obj} in {sub}?"
    elif 'affect' in rel_clean:
        q = f"What affects {sub}?"
    elif 'season' in rel_clean:
        q = f"When is the season for {sub}?"
    elif 'fertiliz' in rel_clean:
        q = f"What fertilizer for {sub}?"
    elif 'plant' in rel_clean:
        q = f"How to plant {sub}?"
    elif 'harvest' in rel_clean:
        q = f"When to harvest {sub}?"
    elif 'pesticid' in rel_clean or 'insecticid' in rel_clean:
        q = f"What pesticide for {sub}?"
    elif 'disease' in rel_clean or 'pest' in rel_clean:
        q = f"What are common {obj} in {sub}?"
    elif 'disease' in obj.lower() or 'pest' in obj.lower():
        q = f"How to control {obj} in {sub}?"
    else:
        if is_crop:
            q = f"How to grow {sub}?"
        else:
            continue
    
    # Add if not duplicate
    if q and q not in generated_questions:
        generated_questions.append(q)
        if len(generated_questions) >= 25:
            break

# Add some generic crop questions
generic_qs = [
    "What is the best fertilizer for rice?",
    "When to sow rice seeds?",
    "How to prevent fungal blast in rice?",
    "What is the nitrogen dose for rice?",
    "How much water does rice need?",
    "What are the rice varieties in Tamil Nadu?",
    "When to apply urea for rice?",
    "How to control pests in cotton?",
    "What is the spacing for sugarcane?",
    "How to control red rot in sugarcane?",
    "What is the fertilizer dose for wheat?",
    "When to harvest maize?",
    "What are the maize varieties?",
    "How to control leaf folder in rice?",
    "What is zinc deficiency in rice?",
]

for q in generic_qs:
    if q not in generated_questions:
        generated_questions.append(q)
    if len(generated_questions) >= 25:
        break

print(f"\nGenerated {len(generated_questions)} questions:")
for i, q in enumerate(generated_questions):
    print(f"  {i+1}. {q}")

# Save to ragas_ground_truth.json
ragas_data = {
    "dataset_name": "Agri-RAG RAGAS Evaluation Dataset",
    "description": "Questions extracted from PDF triplets / agriculture knowledge",
    "version": "2.0",
    "questions": []
}

for i, q in enumerate(generated_questions):
    ragas_data["questions"].append({
        "id": i + 1,
        "question": q,
        "ground_truth_answer": "",
        "ground_truth_contexts": []
    })

output_file = Path("data/ragas_ground_truth.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(ragas_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(generated_questions)} questions to: {output_file}")