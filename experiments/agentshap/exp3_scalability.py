"""
Experiment 3: Scalability Test
===============================
Tests how AgentSHAP performance scales with number of tools.
Measures runtime and API calls as tool count increases.

Uses real tools from API-Bank benchmark.
"""

import sys
import os
import json
import time
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

# Import ALL API-Bank tools
from apis.calculator import Calculator
from apis.query_stock import QueryStock
from apis.wiki import Wiki
from apis.query_balance import QueryBalance
from apis.book_hotel import BookHotel
from apis.add_alarm import AddAlarm
from apis.add_reminder import AddReminder
from apis.play_music import PlayMusic
from apis.translate import Translate
from apis.dictionary import Dictionary
from apis.search_engine import SearchEngine
from apis.send_email import SendEmail

# Paths
APIBANK_DIR = EXPERIMENT_DIR / "DAMO-ConvAI" / "api-bank"
DATABASE_DIR = APIBANK_DIR / "init_database"


def load_database(name):
    db_path = DATABASE_DIR / f"{name}.json"
    if db_path.exists():
        with open(db_path) as f:
            return json.load(f)
    return {}


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


def create_tool_pool():
    """Create a pool of diverse tools from API-Bank."""
    tools = []

    # Core tools with databases
    tools.append(apibank_to_tool(Calculator))
    tools.append(apibank_to_tool(QueryStock, database=load_database("Stock")))
    tools.append(apibank_to_tool(Wiki, database=load_database("Wiki")))
    tools.append(apibank_to_tool(QueryBalance, database=load_database("Bank")))
    tools.append(apibank_to_tool(BookHotel, database=load_database("Hotel")))

    # Additional tools
    tools.append(apibank_to_tool(AddAlarm))
    tools.append(apibank_to_tool(AddReminder))
    tools.append(apibank_to_tool(PlayMusic))
    tools.append(apibank_to_tool(Dictionary))
    tools.append(apibank_to_tool(SearchEngine, database=load_database("SearchEngine")))

    return tools


def run_scalability_experiment(api_key, tool_counts=[2, 3, 4, 5, 6, 8, 10], n_runs=3, sampling_ratio=0.5):
    """
    Test AgentSHAP with increasing number of tools.
    Measure runtime and number of API calls.
    """
    model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    all_tools = create_tool_pool()

    # Use a prompt that works with any tool subset
    prompt = "Calculate 25 * 4 for me"

    results = []

    print(f"\n{'='*70}")
    print(f"SCALABILITY EXPERIMENT")
    print(f"{'='*70}")
    print(f"Tool pool size: {len(all_tools)}")
    print(f"Tool counts to test: {tool_counts}")
    print(f"Runs per count: {n_runs}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}\n")

    for n_tools in tool_counts:
        if n_tools > len(all_tools):
            print(f"Skipping {n_tools} tools (only {len(all_tools)} available)")
            continue

        print(f"\n{'#'*70}")
        print(f"TESTING WITH {n_tools} TOOLS")
        print(f"{'#'*70}")

        tools_subset = all_tools[:n_tools]
        print(f"Tools: {[t.name for t in tools_subset]}")

        run_times = []
        api_calls_list = []

        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} ---")

            agent_shap = AgentSHAP(
                model=model,
                tools=tools_subset,
                vectorizer=vectorizer,
                max_iterations=3
            )

            start_time = time.time()
            _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)
            end_time = time.time()

            runtime = end_time - start_time
            run_times.append(runtime)

            # Count combinations tested (from results_df)
            n_combinations = len(agent_shap.results_df) if agent_shap.results_df is not None else 0
            api_calls_list.append(n_combinations + 1)  # +1 for baseline

            print(f"Runtime: {runtime:.2f}s, Combinations: {n_combinations}")

        results.append({
            "n_tools": n_tools,
            "tool_names": [t.name for t in tools_subset],
            "mean_runtime": np.mean(run_times),
            "std_runtime": np.std(run_times),
            "min_runtime": np.min(run_times),
            "max_runtime": np.max(run_times),
            "mean_api_calls": np.mean(api_calls_list),
            "std_api_calls": np.std(api_calls_list),
            "all_runtimes": run_times,
            "all_api_calls": api_calls_list
        })

        print(f"\nMean runtime: {np.mean(run_times):.2f}s ± {np.std(run_times):.2f}s")

    return results


def plot_scalability_results(results, output_path):
    """Create visualization for scalability experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n_tools_list = [r["n_tools"] for r in results]
    mean_runtimes = [r["mean_runtime"] for r in results]
    std_runtimes = [r["std_runtime"] for r in results]
    mean_api_calls = [r["mean_api_calls"] for r in results]

    # Plot 1: Runtime vs Number of Tools
    ax1 = axes[0]
    ax1.errorbar(n_tools_list, mean_runtimes, yerr=std_runtimes, marker='o',
                 capsize=5, color='steelblue', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Tools')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime Scaling')
    ax1.grid(True, alpha=0.3)

    # Plot 2: API Calls vs Number of Tools
    ax2 = axes[1]
    ax2.plot(n_tools_list, mean_api_calls, marker='s', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Tools')
    ax2.set_ylabel('Number of LLM Calls')
    ax2.set_title('API Calls Scaling')
    ax2.grid(True, alpha=0.3)

    # Theoretical 2^n line (for reference)
    theoretical = [2**n for n in n_tools_list]
    ax2.plot(n_tools_list, theoretical, '--', color='red', alpha=0.5, label='2^n (exhaustive)')
    ax2.legend()

    # Plot 3: Runtime per API call
    ax3 = axes[2]
    runtime_per_call = [r["mean_runtime"] / r["mean_api_calls"] for r in results]
    ax3.bar(n_tools_list, runtime_per_call, color='orange', edgecolor='black')
    ax3.set_xlabel('Number of Tools')
    ax3.set_ylabel('Seconds per LLM Call')
    ax3.set_title('Efficiency (Time per Call)')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Experiment 3: Scalability Test\n(API-Bank Benchmark)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_results_to_csv(results, output_dir):
    """Save results to CSV."""
    # Summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            "n_tools": r["n_tools"],
            "mean_runtime": r["mean_runtime"],
            "std_runtime": r["std_runtime"],
            "min_runtime": r["min_runtime"],
            "max_runtime": r["max_runtime"],
            "mean_api_calls": r["mean_api_calls"],
            "tools": ", ".join(r["tool_names"])
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / "exp3_scalability_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Raw data CSV
    raw_rows = []
    for r in results:
        for run_idx, (rt, api) in enumerate(zip(r["all_runtimes"], r["all_api_calls"])):
            raw_rows.append({
                "n_tools": r["n_tools"],
                "run": run_idx + 1,
                "runtime": rt,
                "api_calls": api
            })

    df_raw = pd.DataFrame(raw_rows)
    csv_path = output_dir / "exp3_scalability_raw.csv"
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
    results = run_scalability_experiment(
        api_key=api_key,
        tool_counts=[2, 3, 4, 5, 6, 8, 10],
        n_runs=3,
        sampling_ratio=0.5
    )

    # Print summary
    print("\n" + "="*70)
    print("SCALABILITY SUMMARY")
    print("="*70)
    print(f"\n{'Tools':<8} {'Runtime (s)':<15} {'API Calls':<12}")
    print("-" * 35)
    for r in results:
        print(f"{r['n_tools']:<8} {r['mean_runtime']:.2f} ± {r['std_runtime']:.2f}    {r['mean_api_calls']:.1f}")

    # Save results
    save_results_to_csv(results, results_dir)

    # Plot
    output_path = results_dir / "exp3_scalability.png"
    plot_scalability_results(results, output_path)

    print(f"\n✓ Results saved to: {results_dir}/")
    print("  - exp3_scalability.png")
    print("  - exp3_scalability_summary.csv")
    print("  - exp3_scalability_raw.csv")

    os.system(f"open {output_path}")
