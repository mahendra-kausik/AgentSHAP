# AgentSHAP Experiments

This folder contains experiments for evaluating AgentSHAP on the API-Bank benchmark.

## Setup

### 1. Install dependencies

```bash
pip install openai numpy pandas matplotlib
```

### 2. Get API-Bank benchmark

Clone the DAMO-ConvAI API-Bank repository:

```bash
cd experiments
git clone https://github.com/AlibabaResearch/DAMO-ConvAI.git
```

### 3. Set OpenAI API key

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

Or export directly:

```bash
export OPENAI_API_KEY=your-key-here
```

## Experiments

| Experiment | Description | Script |
|------------|-------------|--------|
| exp1 | Consistency across Monte Carlo runs | `exp1_consistency.py` |
| exp2 | Faithfulness (quality drop when removing tools) | `exp2_faithfulness.py` |
| exp3 | Scalability | `exp3_scalability.py` |
| exp4 | Irrelevant tool injection | `exp4_irrelevant_injection.py` |
| exp5 | Cross-domain attribution | `exp5_cross_domain.py` |
| exp6 | Model comparison | `exp6_model_comparison.py` |

## Running experiments

Run all experiments:

```bash
./run_all.sh
```

Or run individual experiments:

```bash
source .env
python exp1_consistency.py
```

## Results

Results are saved in the `results/` folder as CSV files and PNG visualizations.
