"""
Experiment 6: Model Comparison Test
====================================
Tests AgentSHAP across different LLM backends to demonstrate
model-agnostic applicability.

Compares tool importance rankings across:
- GPT-4o-mini (fast, cheap)
- GPT-4o (stronger reasoning)

Shows that AgentSHAP produces consistent explanations regardless
of the underlying LLM, validating model-agnostic design.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR / "DAMO-ConvAI" / "api-bank"))

# Direct imports
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = load_module('token_shap.base', PROJECT_DIR / 'token_shap' / 'base.py')
tools_mod = load_module('token_shap.tools', PROJECT_DIR / 'token_shap' / 'tools.py')
agent_shap_mod = load_module('token_shap.agent_shap', PROJECT_DIR / 'token_shap' / 'agent_shap.py')

OpenAIModel = base.OpenAIModel
OpenAIEmbeddings = base.OpenAIEmbeddings
Tool = tools_mod.Tool
AgentSHAP = agent_shap_mod.AgentSHAP

# Import API-Bank tools
from apis.calculator import Calculator
from apis.query_stock import QueryStock
from apis.wiki import Wiki

# Paths
APIBANK_DIR = EXPERIMENT_DIR / "DAMO-ConvAI" / "api-bank"
DATABASE_DIR = APIBANK_DIR / "init_database"
SAMPLES_DIR = APIBANK_DIR / "lv1-lv2-samples" / "level-1-given-desc"


def load_database(name):
    db_path = DATABASE_DIR / f"{name}.json"
    if db_path.exists():
        with open(db_path) as f:
            return json.load(f)
    return {}


def load_prompts_from_benchmark(tool_name, max_prompts=2):
    """Load real prompts from API-Bank benchmark."""
    prompts = []
    for jsonl_file in sorted(SAMPLES_DIR.glob(f"{tool_name}-level-1-*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("role") == "User":
                    prompts.append(data["text"])
                    break
        if len(prompts) >= max_prompts:
            break
    return prompts


def apibank_to_tool(api_class, database=None):
    """Convert API-Bank class to AgentSHAP Tool."""
    try:
        if database is not None:
            api_instance = api_class(init_database=database)
        else:
            api_instance = api_class()
    except:
        api_instance = api_class()

    name = api_class.__name__
    description = getattr(api_class, 'description', f'{name} tool')

    properties = {}
    required = []
    input_params = getattr(api_class, 'input_parameters', {})
    for param_name, param_info in input_params.items():
        type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
        json_type = type_map.get(param_info.get("type", "str"), "string")
        properties[param_name] = {
            "type": json_type,
            "description": param_info.get("description", "")
        }
        required.append(param_name)

    definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    def executor(args):
        try:
            result = api_instance.call(**args)
            if result.get("exception"):
                return f"Error: {result['exception']}"
            return str(result.get("output", "OK"))
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(name=name, description=description, definition=definition, executor=executor)


def create_tools():
    """Create tools from API-Bank."""
    tools = []
    tools.append(apibank_to_tool(Calculator))
    tools.append(apibank_to_tool(QueryStock, database=load_database("Stock")))
    tools.append(apibank_to_tool(Wiki, database=load_database("Wiki")))
    return tools


def run_model_comparison(api_key, sampling_ratio=0.5):
    """
    Compare AgentSHAP results across different LLM models.
    """
    # Models to compare
    models = [
        ("gpt-4o-mini", "GPT-4o-mini"),
        ("gpt-4o", "GPT-4o"),
        ("o4-mini", "o4-mini"),
        ("o3", "o3"),
    ]

    # Test prompts (one from each domain)
    test_cases = [
        ("Can you calculate (5+6)*3 for me?", "Calculator"),
        ("What is the stock price of SQ on March 14th, 2022?", "QueryStock"),
        ("Can you help me search artificial intelligence on wikipedia?", "Wiki"),
    ]

    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    tools = create_tools()

    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    print(f"Models: {[m[1] for m in models]}")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Test prompts: {len(test_cases)}")
    print(f"{'='*70}\n")

    results = []

    for model_name, model_label in models:
        print(f"\n{'#'*70}")
        print(f"MODEL: {model_label} ({model_name})")
        print(f"{'#'*70}")

        model = OpenAIModel(model_name=model_name, api_key=api_key)

        for prompt, expected_tool in test_cases:
            print(f"\n--- Prompt: {prompt[:50]}...")
            print(f"    Expected: {expected_tool}")

            start_time = time.time()

            agent_shap = AgentSHAP(
                model=model,
                tools=tools,
                vectorizer=vectorizer,
                max_iterations=3
            )

            _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)

            elapsed = time.time() - start_time

            # Find top tool
            sorted_shaps = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
            top_tool = sorted_shaps[0][0]

            results.append({
                "model": model_label,
                "model_name": model_name,
                "prompt": prompt,
                "expected_tool": expected_tool,
                "top_tool": top_tool,
                "top_correct": top_tool == expected_tool,
                "shap_values": shap_values,
                "time_seconds": elapsed
            })

            print(f"    SHAP: {shap_values}")
            print(f"    Top: {top_tool} ({'✓' if top_tool == expected_tool else '✗'})")
            print(f"    Time: {elapsed:.1f}s")

    return results, models, test_cases


def plot_model_comparison(results, models, test_cases, output_path):
    """Create visualization comparing models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    tool_names = list(results[0]["shap_values"].keys())
    n_models = len(models)

    # Plot 1: Accuracy per model
    ax1 = axes[0]
    model_accuracy = {}
    for model_name, model_label in models:
        model_results = [r for r in results if r["model"] == model_label]
        correct = sum(1 for r in model_results if r["top_correct"])
        model_accuracy[model_label] = 100 * correct / len(model_results)

    colors = plt.cm.Set2(range(n_models))
    bars = ax1.bar(model_accuracy.keys(), model_accuracy.values(), color=colors, edgecolor='black')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('Tool Attribution Accuracy by Model')
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, model_accuracy.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Plot 2: SHAP correlation between models
    ax2 = axes[1]

    # Get SHAP vectors for each model-prompt pair
    model_shap_vectors = {m[1]: [] for m in models}
    for r in results:
        vec = [r["shap_values"].get(t, 0) for t in tool_names]
        model_shap_vectors[r["model"]].append(vec)

    # Compute correlation between models
    model_labels = [m[1] for m in models]
    if len(model_labels) >= 2:
        vec1 = np.array(model_shap_vectors[model_labels[0]]).flatten()
        vec2 = np.array(model_shap_vectors[model_labels[1]]).flatten()

        correlation = np.corrcoef(vec1, vec2)[0, 1]

        ax2.scatter(vec1, vec2, c='steelblue', s=100, edgecolor='black', alpha=0.7)
        ax2.plot([0, max(max(vec1), max(vec2))], [0, max(max(vec1), max(vec2))],
                 'r--', label='Perfect agreement')
        ax2.set_xlabel(f'{model_labels[0]} SHAP')
        ax2.set_ylabel(f'{model_labels[1]} SHAP')
        ax2.set_title(f'SHAP Value Correlation\nr = {correlation:.3f}')
        ax2.legend()

    # Plot 3: Time comparison
    ax3 = axes[2]
    model_times = {}
    for model_name, model_label in models:
        model_results = [r for r in results if r["model"] == model_label]
        model_times[model_label] = np.mean([r["time_seconds"] for r in model_results])

    bars = ax3.bar(model_times.keys(), model_times.values(), color=colors, edgecolor='black')
    ax3.set_ylabel('Mean Time (seconds)')
    ax3.set_title('Analysis Time by Model')
    for bar, val in zip(bars, model_times.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', fontsize=10)

    plt.suptitle('Experiment 6: Model Comparison\nAgentSHAP is Model-Agnostic',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_results_to_csv(results, output_dir):
    """Save results to CSV."""
    # Summary
    rows = []
    for r in results:
        row = {
            "model": r["model"],
            "prompt": r["prompt"][:50],
            "expected_tool": r["expected_tool"],
            "top_tool": r["top_tool"],
            "top_correct": r["top_correct"],
            "time_seconds": r["time_seconds"]
        }
        for tool, shap in r["shap_values"].items():
            row[f"shap_{tool}"] = shap
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "exp6_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Run experiment
    results, models, test_cases = run_model_comparison(api_key=api_key, sampling_ratio=0.5)

    # Print summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    for model_name, model_label in models:
        model_results = [r for r in results if r["model"] == model_label]
        correct = sum(1 for r in model_results if r["top_correct"])
        mean_time = np.mean([r["time_seconds"] for r in model_results])
        print(f"\n{model_label}:")
        print(f"  Accuracy: {correct}/{len(model_results)} ({100*correct/len(model_results):.0f}%)")
        print(f"  Mean time: {mean_time:.1f}s")

    # Save results
    save_results_to_csv(results, results_dir)

    # Plot
    output_path = results_dir / "exp6_model_comparison.png"
    plot_model_comparison(results, models, test_cases, output_path)

    print(f"\n✓ Results saved to: {results_dir}/")
    print("  - exp6_model_comparison.png")
    print("  - exp6_model_comparison.csv")

    os.system(f"open {output_path}")
