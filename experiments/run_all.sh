#!/bin/bash
# Run all AgentSHAP experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load API key
source "$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"

echo "=========================================="
echo "Running ALL AgentSHAP Experiments"
echo "Using OpenAI Embeddings (text-embedding-3-large)"
echo "=========================================="

# Experiment 1: Consistency
echo ""
echo ">>> Experiment 1: Consistency Test"
python exp1_consistency.py

# Experiment 2: Faithfulness
echo ""
echo ">>> Experiment 2: Faithfulness Test"
python exp2_faithfulness.py

# Experiment 3: Scalability
echo ""
echo ">>> Experiment 3: Scalability Test"
python exp3_scalability.py

# Experiment 4: Irrelevant Tool Injection
echo ""
echo ">>> Experiment 4: Irrelevant Tool Injection Test"
python exp4_irrelevant_injection.py

# Experiment 5: Cross-Domain Queries
echo ""
echo ">>> Experiment 5: Cross-Domain Query Test"
python exp5_cross_domain.py

# Experiment 6: Model Comparison
echo ""
echo ">>> Experiment 6: Model Comparison (GPT-4o-mini vs GPT-4o)"
python exp6_model_comparison.py

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $SCRIPT_DIR/results/"
echo "=========================================="
echo ""
echo "Files generated:"
ls -la "$SCRIPT_DIR/results/"
