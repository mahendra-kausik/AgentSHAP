"""
Experiment 4: Irrelevant Tool Injection Test
=============================================
Similar to TokenSHAP's "Injection of Random Words" experiment.

Tests if AgentSHAP correctly assigns low SHAP values to irrelevant/decoy
tools that are injected into the tool set.

Uses real tools from API-Bank benchmark.
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
from apis.play_music import PlayMusic
from apis.book_hotel import BookHotel
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


def load_prompts_from_benchmark(tool_name, max_prompts=5):
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


def create_relevant_tools():
    """Create the relevant tools (Calculator, Stock, Wiki)."""
    tools = []
    tools.append(apibank_to_tool(Calculator))
    tools.append(apibank_to_tool(QueryStock, database=load_database("Stock")))
    tools.append(apibank_to_tool(Wiki, database=load_database("Wiki")))
    return tools


def create_irrelevant_tools():
    """Create irrelevant/decoy tools that should get low SHAP values."""
    tools = []
    tools.append(apibank_to_tool(AddAlarm))
    tools.append(apibank_to_tool(AddReminder))
    tools.append(apibank_to_tool(PlayMusic))
    tools.append(apibank_to_tool(BookHotel, database=load_database("Hotel")))
    return tools


def run_injection_experiment(api_key, sampling_ratio=0.5):
    """
    For each prompt:
    1. Run AgentSHAP with relevant tools only
    2. Run AgentSHAP with relevant + irrelevant tools injected
    3. Compare SHAP values: irrelevant tools should have low SHAP

    Similar to TokenSHAP's random word injection experiment.
    """
    model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")

    relevant_tools = create_relevant_tools()
    irrelevant_tools = create_irrelevant_tools()
    all_tools = relevant_tools + irrelevant_tools

    relevant_names = {t.name for t in relevant_tools}
    irrelevant_names = {t.name for t in irrelevant_tools}

    # Load prompts from benchmark
    prompts = []
    expected_tools = []

    calc_prompts = load_prompts_from_benchmark("Calculator", max_prompts=3)
    prompts.extend(calc_prompts)
    expected_tools.extend(["Calculator"] * len(calc_prompts))

    stock_prompts = load_prompts_from_benchmark("QueryStock", max_prompts=3)
    prompts.extend(stock_prompts)
    expected_tools.extend(["QueryStock"] * len(stock_prompts))

    wiki_prompts = load_prompts_from_benchmark("Wiki", max_prompts=1)
    prompts.extend(wiki_prompts)
    expected_tools.extend(["Wiki"] * len(wiki_prompts))

    print(f"\n{'='*70}")
    print(f"IRRELEVANT TOOL INJECTION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Relevant tools: {[t.name for t in relevant_tools]}")
    print(f"Irrelevant (injected) tools: {[t.name for t in irrelevant_tools]}")
    print(f"Total tools: {len(all_tools)}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*70}\n")

    results = []

    for idx, (prompt, expected) in enumerate(zip(prompts, expected_tools)):
        print(f"\n{'#'*70}")
        print(f"PROMPT {idx+1}/{len(prompts)}: {prompt[:60]}...")
        print(f"Expected relevant tool: {expected}")
        print(f"{'#'*70}")

        # Run with all tools (relevant + irrelevant)
        agent_shap = AgentSHAP(
            model=model,
            tools=all_tools,
            vectorizer=vectorizer,
            max_iterations=3
        )
        _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)

        # Separate SHAP values into relevant vs irrelevant
        relevant_shaps = {k: v for k, v in shap_values.items() if k in relevant_names}
        irrelevant_shaps = {k: v for k, v in shap_values.items() if k in irrelevant_names}

        print(f"\nSHAP Values:")
        print(f"  Relevant tools: {relevant_shaps}")
        print(f"  Irrelevant tools: {irrelevant_shaps}")

        # Calculate statistics
        relevant_values = list(relevant_shaps.values())
        irrelevant_values = list(irrelevant_shaps.values())

        mean_relevant = np.mean(relevant_values) if relevant_values else 0
        mean_irrelevant = np.mean(irrelevant_values) if irrelevant_values else 0
        diff = mean_relevant - mean_irrelevant

        # Check if top tool is the expected one
        top_tool = max(shap_values.items(), key=lambda x: x[1])[0]
        top_correct = top_tool == expected

        results.append({
            "prompt": prompt,
            "expected_tool": expected,
            "top_tool": top_tool,
            "top_correct": top_correct,
            "shap_values": shap_values,
            "relevant_shaps": relevant_shaps,
            "irrelevant_shaps": irrelevant_shaps,
            "mean_relevant": mean_relevant,
            "mean_irrelevant": mean_irrelevant,
            "mean_diff": diff,
            "max_irrelevant": max(irrelevant_values) if irrelevant_values else 0,
            "all_irrelevant_low": all(v < 0.1 for v in irrelevant_values)
        })

        print(f"\nMean SHAP (relevant): {mean_relevant:.4f}")
        print(f"Mean SHAP (irrelevant): {mean_irrelevant:.4f}")
        print(f"Difference: {diff:.4f}")
        print(f"Top tool correct: {top_correct}")

    return results


def plot_injection_results(results, output_path):
    """Create visualization similar to TokenSHAP Figure 3-5."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect all SHAP values
    all_relevant = []
    all_irrelevant = []
    for r in results:
        all_relevant.extend(r["relevant_shaps"].values())
        all_irrelevant.extend(r["irrelevant_shaps"].values())

    # Plot 1: Boxplot comparison (like TokenSHAP Figures 3-5)
    ax1 = axes[0, 0]
    bp = ax1.boxplot([all_relevant, all_irrelevant],
                      labels=['Relevant Tools', 'Irrelevant Tools'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax1.set_ylabel('SHAP Value')
    ax1.set_title('SHAP Distribution: Relevant vs Irrelevant Tools')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Mean difference per prompt
    ax2 = axes[0, 1]
    prompt_labels = [f"P{i+1}" for i in range(len(results))]
    diffs = [r["mean_diff"] for r in results]
    colors = ['green' if d > 0 else 'red' for d in diffs]
    bars = ax2.bar(prompt_labels, diffs, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_ylabel('Mean SHAP Difference (Relevant - Irrelevant)')
    ax2.set_title('SHAP Difference per Prompt')
    ax2.set_xlabel('Prompt')

    # Plot 3: Top-1 accuracy
    ax3 = axes[1, 0]
    correct = sum(1 for r in results if r["top_correct"])
    incorrect = len(results) - correct
    ax3.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
            colors=['green', 'red'], autopct='%1.0f%%', startangle=90)
    ax3.set_title(f'Top-1 Tool Accuracy\n({correct}/{len(results)} prompts)')

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    INJECTION EXPERIMENT SUMMARY
    ════════════════════════════════════════

    Data Source: API-Bank Benchmark

    Relevant tools: Calculator, QueryStock, Wiki
    Irrelevant tools: AddAlarm, AddReminder, PlayMusic, BookHotel

    RESULTS:
    • Mean SHAP (relevant): {np.mean(all_relevant):.4f}
    • Mean SHAP (irrelevant): {np.mean(all_irrelevant):.4f}
    • Mean Difference: {np.mean(diffs):.4f}
    • Std Difference: {np.std(diffs):.4f}

    • Top-1 Accuracy: {100*correct/len(results):.0f}%
    • All irrelevant < 0.1: {sum(1 for r in results if r['all_irrelevant_low'])}/{len(results)}

    CONCLUSION:
    AgentSHAP {"correctly" if np.mean(diffs) > 0 else "incorrectly"}
    assigns {"higher" if np.mean(diffs) > 0 else "lower"} importance
    to relevant tools vs irrelevant injected tools.
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 4: Irrelevant Tool Injection\n(API-Bank Benchmark)',
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
            "prompt": r["prompt"],
            "expected_tool": r["expected_tool"],
            "top_tool": r["top_tool"],
            "top_correct": r["top_correct"],
            "mean_relevant": r["mean_relevant"],
            "mean_irrelevant": r["mean_irrelevant"],
            "mean_diff": r["mean_diff"],
            "max_irrelevant": r["max_irrelevant"],
            "all_irrelevant_low": r["all_irrelevant_low"]
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / "exp4_injection_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Raw SHAP values per tool per prompt
    raw_rows = []
    for r in results:
        for tool, shap in r["shap_values"].items():
            raw_rows.append({
                "prompt": r["prompt"][:50],
                "tool": tool,
                "shap_value": shap,
                "is_relevant": tool in r["relevant_shaps"],
                "expected_tool": r["expected_tool"]
            })

    df_raw = pd.DataFrame(raw_rows)
    csv_path = output_dir / "exp4_injection_raw.csv"
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
    results = run_injection_experiment(api_key=api_key, sampling_ratio=0.5)

    # Print summary
    print("\n" + "="*70)
    print("INJECTION EXPERIMENT SUMMARY")
    print("="*70)

    all_diffs = [r["mean_diff"] for r in results]
    print(f"\nMean SHAP difference (relevant - irrelevant): {np.mean(all_diffs):.4f} ± {np.std(all_diffs):.4f}")
    print(f"Top-1 accuracy: {sum(1 for r in results if r['top_correct'])}/{len(results)}")

    # Save results
    save_results_to_csv(results, results_dir)

    # Plot
    output_path = results_dir / "exp4_injection.png"
    plot_injection_results(results, output_path)

    print(f"\n✓ Results saved to: {results_dir}/")
    print("  - exp4_injection.png")
    print("  - exp4_injection_summary.csv")
    print("  - exp4_injection_raw.csv")

    os.system(f"open {output_path}")
