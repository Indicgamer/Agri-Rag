"""
Analyze evaluation results
"""
import json

with open('evaluation_results.json') as f:
    data = json.load(f)

results = data['results']
metrics = data['metrics']

print('='*60)
print('EVALUATION RESULTS ANALYSIS')
print('='*60)

# Calculate averages
nli_wins = 0
baseline_wins = 0
ties = 0
total_hallucination_reduction = 0

for r in results:
    hr = r.get('hallucination_reduction', 0)
    total_hallucination_reduction += hr
    
    if hr > 0.01:
        nli_wins += 1
    elif hr < -0.01:
        baseline_wins += 1
    else:
        ties += 1

n = len(results)
print(f'\nTotal Questions: {n}')

print(f'\n--- HALLUCINATION REDUCTION ---')
print(f'NLI wins: {nli_wins}/{n} ({nli_wins/n*100:.1f}%)')
print(f'Baseline wins: {baseline_wins}/{n} ({baseline_wins/n*100:.1f}%)')
print(f'Ties: {ties}/{n} ({ties/n*100:.1f}%)')
print(f'Avg hallucination reduction: {total_hallucination_reduction/n:.3f}')

print(f'\n--- RAGAS METRICS ---')
print(f'NLI Faithfulness: {metrics.get("nli_faithfulness", 0):.3f}')
print(f'Baseline Faithfulness: {metrics.get("baseline_faithfulness", 0):.3f}')
print(f'NLI Avg Confidence: {metrics.get("nli_confidence", 0):.3f}')
print(f'Baseline Avg Confidence: {metrics.get("baseline_confidence", 0):.3f}')

# Show questions where NLI won
print(f'\n--- QUESTIONS WHERE NLI WON ---')
for r in results:
    if r.get('hallucination_reduction', 0) > 0.01:
        print(f'  + {r["question"][:55]}')
        print(f'    reduction: {r["hallucination_reduction"]:.2f}, facts used: {r["nli_facts_used"]}/{r["baseline_facts_used"]}')

print(f'\n--- QUESTIONS WHERE BASELINE WON ---')
for r in results:
    if r.get('hallucination_reduction', 0) < -0.01:
        print(f'  - {r["question"][:55]}')
        print(f'    reduction: {r["hallucination_reduction"]:.2f}, facts used: {r["nli_facts_used"]}/{r["baseline_facts_used"]}')