"""
Experiment 5: Cross-Domain Query Test
======================================
Tests AgentSHAP's ability to correctly identify the relevant tool
when queries span different domains (math, finance, knowledge).

This is the key validation: can AgentSHAP distinguish which tool
is most important for each domain-specific query?

Uses real tools and prompts from API-Bank benchmark.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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
from apis.add_alarm import AddAlarm
from apis.add_reminder import AddReminder
from apis.translate import Translate

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


def load_prompts_from_benchmark(tool_name, max_prompts=3):
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
    """Create diverse set of tools for cross-domain testing."""
    tools = []
    tools.append(apibank_to_tool(Calculator))
    tools.append(apibank_to_tool(QueryStock, database=load_database("Stock")))
    tools.append(apibank_to_tool(Wiki, database=load_database("Wiki")))
    tools.append(apibank_to_tool(AddAlarm))
    tools.append(apibank_to_tool(AddReminder))
    tools.append(apibank_to_tool(Translate))
    return tools


def run_cross_domain_experiment(api_key, sampling_ratio=0.5):
    """
    Test AgentSHAP across different query domains:
    - Math domain (Calculator)
    - Finance domain (QueryStock)
    - Knowledge domain (Wiki)

    Validates that SHAP correctly identifies the relevant tool for each domain.
    """
    model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    tools = create_tools()

    # Define domains with their expected tools
    domains = {
        "Math": {"tool": "Calculator", "prompts": load_prompts_from_benchmark("Calculator", 3)},
        "Finance": {"tool": "QueryStock", "prompts": load_prompts_from_benchmark("QueryStock", 3)},
        "Knowledge": {"tool": "Wiki", "prompts": load_prompts_from_benchmark("Wiki", 2)},
    }

    print(f"\n{'='*70}")
    print(f"CROSS-DOMAIN QUERY TEST")
    print(f"{'='*70}")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Domains: {list(domains.keys())}")
    for domain, info in domains.items():
        print(f"  {domain}: {len(info['prompts'])} prompts, expected tool: {info['tool']}")
    print(f"{'='*70}\n")

    results = []

    for domain_name, domain_info in domains.items():
        expected_tool = domain_info["tool"]
        prompts = domain_info["prompts"]

        for idx, prompt in enumerate(prompts):
            print(f"\n{'#'*70}")
            print(f"DOMAIN: {domain_name} | PROMPT {idx+1}/{len(prompts)}")
            print(f"Prompt: {prompt[:60]}...")
            print(f"Expected tool: {expected_tool}")
            print(f"{'#'*70}")

            agent_shap = AgentSHAP(
                model=model,
                tools=tools,
                vectorizer=vectorizer,
                max_iterations=3
            )

            _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)

            # Find top tool
            sorted_shaps = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
            top_tool = sorted_shaps[0][0]
            top_shap = sorted_shaps[0][1]

            # Get SHAP value for expected tool
            expected_shap = shap_values.get(expected_tool, 0)
            expected_rank = next((i+1 for i, (t, _) in enumerate(sorted_shaps) if t == expected_tool), -1)

            results.append({
                "domain": domain_name,
                "prompt": prompt,
                "expected_tool": expected_tool,
                "top_tool": top_tool,
                "top_correct": top_tool == expected_tool,
                "top_shap": top_shap,
                "expected_tool_shap": expected_shap,
                "expected_tool_rank": expected_rank,
                "shap_values": shap_values
            })

            print(f"\nSHAP Values: {shap_values}")
            print(f"Top tool: {top_tool} (SHAP={top_shap:.4f})")
            print(f"Expected tool rank: {expected_rank}")
            print(f"Correct: {top_tool == expected_tool}")

    return results, domains


def plot_cross_domain_results(results, domains, output_path):
    """Create visualization for cross-domain experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy per domain
    ax1 = axes[0, 0]
    domain_accuracy = {}
    for domain in domains.keys():
        domain_results = [r for r in results if r["domain"] == domain]
        correct = sum(1 for r in domain_results if r["top_correct"])
        domain_accuracy[domain] = 100 * correct / len(domain_results) if domain_results else 0

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax1.bar(domain_accuracy.keys(), domain_accuracy.values(), color=colors, edgecolor='black')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('Tool Selection Accuracy by Domain')
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, domain_accuracy.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Plot 2: Expected tool rank distribution
    ax2 = axes[0, 1]
    ranks = [r["expected_tool_rank"] for r in results]
    unique_ranks = sorted(set(ranks))
    rank_counts = [ranks.count(r) for r in unique_ranks]
    ax2.bar([f"Rank {r}" for r in unique_ranks], rank_counts, color='steelblue', edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Expected Tool Ranking Distribution')
    ax2.set_xlabel('Rank of Expected Tool')

    # Plot 3: SHAP values heatmap per domain
    ax3 = axes[1, 0]

    # Get all tool names
    all_tools = sorted(set(t for r in results for t in r["shap_values"].keys()))
    domain_names = list(domains.keys())

    # Calculate mean SHAP per tool per domain
    heatmap_data = []
    for domain in domain_names:
        domain_results = [r for r in results if r["domain"] == domain]
        domain_means = []
        for tool in all_tools:
            values = [r["shap_values"].get(tool, 0) for r in domain_results]
            domain_means.append(np.mean(values) if values else 0)
        heatmap_data.append(domain_means)

    heatmap = ax3.imshow(heatmap_data, cmap='coolwarm', aspect='auto')
    ax3.set_xticks(range(len(all_tools)))
    ax3.set_xticklabels(all_tools, rotation=45, ha='right', fontsize=9)
    ax3.set_yticks(range(len(domain_names)))
    ax3.set_yticklabels(domain_names)
    ax3.set_title('Mean SHAP Values by Domain')

    # Add values to heatmap
    for i in range(len(domain_names)):
        for j in range(len(all_tools)):
            ax3.text(j, i, f'{heatmap_data[i][j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if heatmap_data[i][j] > 0.3 else 'black')

    plt.colorbar(heatmap, ax=ax3, label='Mean SHAP')

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    overall_accuracy = 100 * sum(1 for r in results if r["top_correct"]) / len(results)
    mean_expected_rank = np.mean([r["expected_tool_rank"] for r in results])

    summary_text = f"""
    CROSS-DOMAIN EXPERIMENT SUMMARY
    ════════════════════════════════════════

    Data Source: API-Bank Benchmark

    Domains Tested: {len(domains)}
    Total Prompts: {len(results)}

    RESULTS BY DOMAIN:
    • Math (Calculator): {domain_accuracy.get('Math', 0):.0f}% accuracy
    • Finance (QueryStock): {domain_accuracy.get('Finance', 0):.0f}% accuracy
    • Knowledge (Wiki): {domain_accuracy.get('Knowledge', 0):.0f}% accuracy

    OVERALL RESULTS:
    • Top-1 Accuracy: {overall_accuracy:.0f}%
    • Mean Expected Tool Rank: {mean_expected_rank:.2f}

    CONCLUSION:
    AgentSHAP {"successfully" if overall_accuracy >= 50 else "partially"}
    identifies domain-relevant tools across
    different query types.
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 5: Cross-Domain Query Test\n(API-Bank Benchmark)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_results_to_csv(results, output_dir):
    """Save results to CSV."""
    # Summary per prompt
    summary_rows = []
    for r in results:
        summary_rows.append({
            "domain": r["domain"],
            "prompt": r["prompt"],
            "expected_tool": r["expected_tool"],
            "top_tool": r["top_tool"],
            "top_correct": r["top_correct"],
            "expected_tool_shap": r["expected_tool_shap"],
            "expected_tool_rank": r["expected_tool_rank"],
            "top_shap": r["top_shap"]
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / "exp5_cross_domain_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Raw SHAP values
    raw_rows = []
    for r in results:
        for tool, shap in r["shap_values"].items():
            raw_rows.append({
                "domain": r["domain"],
                "prompt": r["prompt"][:50],
                "tool": tool,
                "shap_value": shap,
                "is_expected": tool == r["expected_tool"]
            })

    df_raw = pd.DataFrame(raw_rows)
    csv_path = output_dir / "exp5_cross_domain_raw.csv"
    df_raw.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df_summary, df_raw


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Run experiment
    results, domains = run_cross_domain_experiment(api_key=api_key, sampling_ratio=0.5)

    # Print summary
    print("\n" + "="*70)
    print("CROSS-DOMAIN SUMMARY")
    print("="*70)

    for domain in domains.keys():
        domain_results = [r for r in results if r["domain"] == domain]
        correct = sum(1 for r in domain_results if r["top_correct"])
        print(f"\n{domain}:")
        print(f"  Accuracy: {correct}/{len(domain_results)} ({100*correct/len(domain_results):.0f}%)")

    overall_correct = sum(1 for r in results if r["top_correct"])
    print(f"\nOverall Top-1 Accuracy: {overall_correct}/{len(results)} ({100*overall_correct/len(results):.0f}%)")

    # Save results
    save_results_to_csv(results, results_dir)

    # Plot
    output_path = results_dir / "exp5_cross_domain.png"
    plot_cross_domain_results(results, domains, output_path)

    print(f"\n✓ Results saved to: {results_dir}/")
    print("  - exp5_cross_domain.png")
    print("  - exp5_cross_domain_summary.csv")
    print("  - exp5_cross_domain_raw.csv")

    os.system(f"open {output_path}")
