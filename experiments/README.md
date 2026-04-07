# AgentSHAP Experiments

This folder contains experiments for evaluating AgentSHAP on the API-Bank benchmark.

## Setup

### 1. Install dependencies

```bash
pip install openai sentence-transformers numpy pandas matplotlib
```

### 1.5 Start Ollama (default provider)

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

### 2. Get API-Bank benchmark

Clone the DAMO-ConvAI API-Bank repository:

```bash
cd experiments
git clone https://github.com/AlibabaResearch/DAMO-ConvAI.git
```

### 3. Configure provider via `.env`

Create a `.env` file:

```bash
echo "MODEL_PROVIDER=ollama" > .env
echo "OLLAMA_MODEL_NAME=qwen2.5:7b-instruct" >> .env
echo "OLLAMA_API_URL=http://localhost:11434" >> .env
```

Optional: use an OpenAI-compatible cloud endpoint instead:

```bash
echo "MODEL_PROVIDER=openai_compat" >> .env
echo "OPENAI_COMPAT_API_KEY=your-key-here" >> .env
echo "OPENAI_COMPAT_MODEL_NAME=gemini-2.5-flash" >> .env
echo "OPENAI_COMPAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/" >> .env
```

## Experiments

| Experiment | Description | Script |
|------------|-------------|--------|
| exp1 | Consistency across Monte Carlo runs | `exp1_consistency.py` |
| exp2 | Faithfulness (quality drop when removing tools) | `exp2_faithfulness.py` |
| exp3 | Scalability | `exp3_scalability.py` |
| exp4 | Irrelevant tool injection | `exp4_irrelevant_injection.py` |
| exp5 | Cross-domain attribution | `exp5_cross_domain.py` |

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
