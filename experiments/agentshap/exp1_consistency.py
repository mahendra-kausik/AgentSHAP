"""
Experiment 1: Consistency Test
==============================
Tests if AgentSHAP produces stable SHAP values across multiple runs
with the same agent, tools, and prompt.

Uses real tools from API-Bank benchmark.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR / "DAMO-ConvAI" / "api-bank"))

# Direct imports to avoid cv2 dependency
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

# Load databases
def load_database(name):
    db_path = DATABASE_DIR / f"{name}.json"
    with open(db_path) as f:
        return json.load(f)

def load_prompts_from_benchmark(tool_name, max_prompts=5):
    """Load real prompts from API-Bank benchmark for a given tool."""
    prompts = []
    for jsonl_file in SAMPLES_DIR.glob(f"{tool_name}-level-1-*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("role") == "User":
                    prompts.append(data["text"])
                    break  # One prompt per file
        if len(prompts) >= max_prompts:
            break
    return prompts

def apibank_to_tool(api_class, database=None):
    """
    Convert an API-Bank API class to an AgentSHAP Tool.
    Extracts name, description, and parameters directly from the benchmark.
    """
    # Instantiate the API
    if database is not None:
        api_instance = api_class(init_database=database)
    else:
        api_instance = api_class()

    # Get name from class name (e.g., "Calculator" -> "calculator")
    name = api_class.__name__

    # Get description directly from API-Bank
    description = api_class.description

    # Convert API-Bank input_parameters to OpenAI function schema
    properties = {}
    required = []
    for param_name, param_info in api_class.input_parameters.items():
        # Map API-Bank types to JSON schema types
        type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
        json_type = type_map.get(param_info.get("type", "str"), "string")

        properties[param_name] = {
            "type": json_type,
            "description": param_info.get("description", "")
        }
        required.append(param_name)

    # Build OpenAI function definition
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

    # Create executor that calls the real API-Bank implementation
    def executor(args):
        # Call the API with the provided arguments
        result = api_instance.call(**args)
        if result["exception"]:
            return f"Error: {result['exception']}"
        return str(result["output"])

    return Tool(
        name=name,
        description=description,
        definition=definition,
        executor=executor
    )


def create_tools():
    """Create tools from API-Bank with real databases - all from benchmark data."""

    tools = []

    # Calculator (no database needed)
    tools.append(apibank_to_tool(Calculator))

    # Stock Query (with real database from benchmark)
    stock_db = load_database("Stock")
    tools.append(apibank_to_tool(QueryStock, database=stock_db))

    # Wiki (with real database from benchmark)
    wiki_db = load_database("Wiki")
    tools.append(apibank_to_tool(Wiki, database=wiki_db))

    return tools


def run_consistency_experiment(api_key, prompt, n_runs=5, sampling_ratio=0.5):
    """
    Run AgentSHAP multiple times and measure consistency.
    """
    model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    tools = create_tools()

    print(f"\n{'='*70}")
    print(f"CONSISTENCY EXPERIMENT")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Runs: {n_runs}")
    print(f"{'='*70}\n")

    all_shap_values = []
    all_rankings = []

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")

        agent_shap = AgentSHAP(
            model=model,
            tools=tools,
            vectorizer=vectorizer,
            max_iterations=3
        )

        _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)

        all_shap_values.append(shap_values)

        # Get ranking
        ranking = sorted(shap_values.keys(), key=lambda x: shap_values[x], reverse=True)
        all_rankings.append(ranking)

        print(f"SHAP: {shap_values}")
        print(f"Ranking: {ranking}")

    return all_shap_values, all_rankings, [t.name for t in tools]


def calculate_metrics(all_shap_values, all_rankings, tool_names):
    """Calculate consistency metrics."""
    n_runs = len(all_shap_values)

    # Convert to matrix
    shap_matrix = np.array([[shap[t] for t in tool_names] for shap in all_shap_values])

    # Pairwise cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    similarities = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            sim = cosine_sim(shap_matrix[i], shap_matrix[j])
            similarities.append(sim)

    # Top-K consistency
    top1_tools = [r[0] for r in all_rankings]
    top1_consistency = len(set(top1_tools)) == 1

    top2_sets = [set(r[:2]) for r in all_rankings]
    top2_consistency = all(s == top2_sets[0] for s in top2_sets)

    # Per-tool statistics
    tool_stats = {}
    for i, tool in enumerate(tool_names):
        vals = shap_matrix[:, i]
        tool_stats[tool] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "cv": np.std(vals) / (np.mean(vals) + 1e-10)
        }

    return {
        "cosine_similarity_mean": np.mean(similarities),
        "cosine_similarity_min": np.min(similarities),
        "cosine_similarity_max": np.max(similarities),
        "top1_consistent": top1_consistency,
        "top2_consistent": top2_consistency,
        "top1_tool": max(set(top1_tools), key=top1_tools.count),
        "tool_stats": tool_stats,
        "shap_matrix": shap_matrix
    }


def plot_results(metrics, tool_names, output_path):
    """Create visualization and save to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: SHAP values distribution per tool
    ax1 = axes[0]
    shap_matrix = metrics["shap_matrix"]

    positions = np.arange(len(tool_names))
    bp = ax1.boxplot([shap_matrix[:, i] for i in range(len(tool_names))],
                      positions=positions, widths=0.6, patch_artist=True)

    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(tool_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(tool_names, rotation=45, ha='right')
    ax1.set_ylabel('SHAP Value')
    ax1.set_title('SHAP Value Distribution Across Runs')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Metrics summary
    ax2 = axes[1]
    metric_names = ['Cosine Sim\n(mean)', 'Top-1\nConsistent', 'Top-2\nConsistent']
    metric_values = [
        metrics["cosine_similarity_mean"],
        1.0 if metrics["top1_consistent"] else 0.0,
        1.0 if metrics["top2_consistent"] else 0.0
    ]

    bars = ax2.bar(metric_names, metric_values, color=['steelblue', 'green' if metrics["top1_consistent"] else 'red',
                                                        'green' if metrics["top2_consistent"] else 'red'])
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score')
    ax2.set_title('Consistency Metrics')

    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)

    plt.suptitle(f'AgentSHAP Consistency Test\nTop-1 Tool: {metrics["top1_tool"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def print_results(metrics, tool_names):
    """Print results summary."""
    print("\n" + "="*70)
    print("CONSISTENCY TEST RESULTS")
    print("="*70)

    print(f"\n📊 Cosine Similarity (pairwise):")
    print(f"   Mean: {metrics['cosine_similarity_mean']:.4f}")
    print(f"   Min:  {metrics['cosine_similarity_min']:.4f}")
    print(f"   Max:  {metrics['cosine_similarity_max']:.4f}")

    print(f"\n🎯 Ranking Consistency:")
    print(f"   Top-1 Consistent: {'✓ YES' if metrics['top1_consistent'] else '✗ NO'}")
    print(f"   Top-2 Consistent: {'✓ YES' if metrics['top2_consistent'] else '✗ NO'}")
    print(f"   Most Important Tool: {metrics['top1_tool']}")

    print(f"\n📈 Per-Tool Statistics:")
    print(f"   {'Tool':<15} {'Mean':>10} {'Std':>10} {'CV':>10}")
    print(f"   {'-'*45}")
    for tool in tool_names:
        stats = metrics['tool_stats'][tool]
        print(f"   {tool:<15} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['cv']:>10.2%}")

    print("\n" + "="*70)


def run_multi_prompt_experiment(api_key, n_runs_per_prompt=3, sampling_ratio=0.5):
    """
    Run consistency experiment on multiple real prompts from benchmark.
    """
    # Load real prompts from benchmark
    prompts = []
    prompt_sources = []

    # Get ALL Calculator prompts (3 available)
    calc_prompts = load_prompts_from_benchmark("Calculator", max_prompts=10)
    prompts.extend(calc_prompts)
    prompt_sources.extend(["Calculator"] * len(calc_prompts))

    # Get ALL QueryStock prompts (5 available)
    stock_prompts = load_prompts_from_benchmark("QueryStock", max_prompts=10)
    prompts.extend(stock_prompts)
    prompt_sources.extend(["QueryStock"] * len(stock_prompts))

    # Get ALL Wiki prompts (1 available)
    wiki_prompts = load_prompts_from_benchmark("Wiki", max_prompts=10)
    prompts.extend(wiki_prompts)
    prompt_sources.extend(["Wiki"] * len(wiki_prompts))

    print(f"\n{'='*70}")
    print(f"MULTI-PROMPT CONSISTENCY EXPERIMENT")
    print(f"{'='*70}")
    print(f"Prompts from API-Bank benchmark: {len(prompts)}")
    for i, (p, s) in enumerate(zip(prompts, prompt_sources)):
        print(f"  {i+1}. [{s}] {p[:50]}...")
    print(f"Runs per prompt: {n_runs_per_prompt}")
    print(f"{'='*70}\n")

    all_results = []

    for prompt_idx, (prompt, source) in enumerate(zip(prompts, prompt_sources)):
        print(f"\n{'#'*70}")
        print(f"PROMPT {prompt_idx+1}/{len(prompts)}: {prompt}")
        print(f"Expected tool: {source}")
        print(f"{'#'*70}")

        shap_values_list, rankings_list, tool_names = run_consistency_experiment(
            api_key=api_key,
            prompt=prompt,
            n_runs=n_runs_per_prompt,
            sampling_ratio=sampling_ratio
        )

        metrics = calculate_metrics(shap_values_list, rankings_list, tool_names)

        all_results.append({
            "prompt": prompt,
            "expected_tool": source,
            "shap_values_list": shap_values_list,
            "rankings_list": rankings_list,
            "metrics": metrics,
            "tool_names": tool_names
        })

    return all_results


def plot_multi_prompt_results(all_results, output_path):
    """Create comprehensive visualization for multi-prompt experiment."""
    n_prompts = len(all_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cosine similarity across prompts
    ax1 = axes[0, 0]
    similarities = [r["metrics"]["cosine_similarity_mean"] for r in all_results]
    prompt_labels = [f"P{i+1}" for i in range(n_prompts)]
    bars = ax1.bar(prompt_labels, similarities, color='steelblue', edgecolor='black')
    ax1.axhline(y=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('SHAP Vector Consistency per Prompt')
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    for bar, val in zip(bars, similarities):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)

    # Plot 2: Top-1 accuracy (does SHAP correctly identify the expected tool?)
    ax2 = axes[0, 1]
    top1_correct = []
    for r in all_results:
        expected = r["expected_tool"]
        top1_tool = r["metrics"]["top1_tool"]
        top1_correct.append(1 if top1_tool == expected else 0)

    colors = ['green' if c else 'red' for c in top1_correct]
    bars = ax2.bar(prompt_labels, top1_correct, color=colors, edgecolor='black')
    ax2.set_ylabel('Correct (1) / Incorrect (0)')
    ax2.set_title(f'Top-1 Tool Matches Expected\nAccuracy: {np.mean(top1_correct)*100:.0f}%')
    ax2.set_ylim(0, 1.3)

    # Plot 3: SHAP value distributions across all prompts
    ax3 = axes[1, 0]
    tool_names = all_results[0]["tool_names"]

    # Aggregate all SHAP values per tool
    tool_shap_all = {t: [] for t in tool_names}
    for r in all_results:
        for shap_dict in r["shap_values_list"]:
            for t in tool_names:
                tool_shap_all[t].append(shap_dict[t])

    positions = np.arange(len(tool_names))
    bp = ax3.boxplot([tool_shap_all[t] for t in tool_names], positions=positions,
                      widths=0.6, patch_artist=True)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(tool_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(tool_names, rotation=45, ha='right')
    ax3.set_ylabel('SHAP Value')
    ax3.set_title('SHAP Distribution Across All Prompts')

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    EXPERIMENT SUMMARY
    ══════════════════════════════════════

    Data Source: API-Bank Benchmark

    Prompts tested: {n_prompts}
    Runs per prompt: {all_results[0]["metrics"]["shap_matrix"].shape[0]}
    Total runs: {n_prompts * all_results[0]["metrics"]["shap_matrix"].shape[0]}

    CONSISTENCY METRICS:
    • Mean Cosine Similarity: {np.mean(similarities):.3f}
    • Top-1 Accuracy: {np.mean(top1_correct)*100:.0f}%
    • Top-2 Consistent: {sum(r["metrics"]["top2_consistent"] for r in all_results)}/{n_prompts}

    PROMPTS (from benchmark):
    """
    for i, r in enumerate(all_results):
        p = r["prompt"][:40] + "..." if len(r["prompt"]) > 40 else r["prompt"]
        summary_text += f"\n    P{i+1}: {p}"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('AgentSHAP Consistency Experiment\n(Using Real API-Bank Benchmark Data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_results_to_csv(all_results, output_dir):
    """Save raw results to CSV for the paper."""
    import pandas as pd

    # CSV 1: Per-run SHAP values
    rows = []
    for r in all_results:
        for run_idx, shap_dict in enumerate(r["shap_values_list"]):
            row = {
                "prompt": r["prompt"],
                "expected_tool": r["expected_tool"],
                "run": run_idx + 1,
            }
            for tool, val in shap_dict.items():
                row[f"shap_{tool}"] = val
            rows.append(row)

    df_runs = pd.DataFrame(rows)
    csv_path = output_dir / "exp1_raw_shap_values.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # CSV 2: Summary metrics per prompt
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "prompt": r["prompt"],
            "expected_tool": r["expected_tool"],
            "top1_tool": r["metrics"]["top1_tool"],
            "top1_correct": r["metrics"]["top1_tool"] == r["expected_tool"],
            "cosine_sim_mean": r["metrics"]["cosine_similarity_mean"],
            "cosine_sim_min": r["metrics"]["cosine_similarity_min"],
            "cosine_sim_max": r["metrics"]["cosine_similarity_max"],
            "top2_consistent": r["metrics"]["top2_consistent"],
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / "exp1_summary_metrics.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df_runs, df_summary


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Create results directory
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Run multi-prompt experiment with real benchmark data
    all_results = run_multi_prompt_experiment(
        api_key=api_key,
        n_runs_per_prompt=3,
        sampling_ratio=0.5
    )

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for i, r in enumerate(all_results):
        print(f"\nPrompt {i+1}: {r['prompt'][:50]}...")
        print(f"  Expected: {r['expected_tool']}, Got: {r['metrics']['top1_tool']}")
        print(f"  Cosine Sim: {r['metrics']['cosine_similarity_mean']:.3f}")

    # Save CSV data for paper
    save_results_to_csv(all_results, results_dir)

    # Save plot
    output_path = results_dir / "exp1_consistency.png"
    plot_multi_prompt_results(all_results, output_path)

    print(f"\n✓ All results saved to: {results_dir}/")
    print(f"  - exp1_consistency.png (figure)")
    print(f"  - exp1_raw_shap_values.csv (raw data)")
    print(f"  - exp1_summary_metrics.csv (metrics)")

    # Open the plot
    os.system(f"open {output_path}")
