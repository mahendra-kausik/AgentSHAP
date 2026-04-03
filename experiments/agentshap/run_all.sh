#!/bin/bash
# Run all AgentSHAP experiments

# Load API key
source "$(dirname "$0")/.env"

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Running ALL AgentSHAP Experiments"
echo "Using OpenAI Embeddings (text-embedding-3-large)"
echo "=========================================="

# Experiment 1: Consistency
echo ""
echo ">>> Experiment 1: Consistency Test"
python experiments/exp1_consistency.py

# Experiment 2: Faithfulness
echo ""
echo ">>> Experiment 2: Faithfulness Test"
python experiments/exp2_faithfulness.py

# Experiment 3: Scalability
echo ""
echo ">>> Experiment 3: Scalability Test"
python experiments/exp3_scalability.py

# Experiment 4: Irrelevant Tool Injection
echo ""
echo ">>> Experiment 4: Irrelevant Tool Injection Test"
python experiments/exp4_irrelevant_injection.py

# Experiment 5: Cross-Domain Queries
echo ""
echo ">>> Experiment 5: Cross-Domain Query Test"
python experiments/exp5_cross_domain.py

# Experiment 6: Model Comparison
echo ""
echo ">>> Experiment 6: Model Comparison (GPT-4o-mini vs GPT-4o)"
python experiments/exp6_model_comparison.py

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: experiments/results/"
echo "=========================================="
echo ""
echo "Files generated:"
ls -la experiments/results/
