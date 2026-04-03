"""
Experiment 2: Faithfulness Test
================================
Tests if removing high-SHAP tools degrades response quality more than
removing low-SHAP tools. This validates that SHAP values are faithful
to actual tool importance.

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


def load_database(name):
    db_path = DATABASE_DIR / f"{name}.json"
    with open(db_path) as f:
        return json.load(f)


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
    if database is not None:
        api_instance = api_class(init_database=database)
    else:
        api_instance = api_class()

    name = api_class.__name__
    description = api_class.description

    properties = {}
    required = []
    for param_name, param_info in api_class.input_parameters.items():
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
        result = api_instance.call(**args)
        if result["exception"]:
            return f"Error: {result['exception']}"
        return str(result["output"])

    return Tool(name=name, description=description, definition=definition, executor=executor)


def create_tools():
    """Create tools from API-Bank."""
    tools = []
    tools.append(apibank_to_tool(Calculator))
    stock_db = load_database("Stock")
    tools.append(apibank_to_tool(QueryStock, database=stock_db))
    wiki_db = load_database("Wiki")
    tools.append(apibank_to_tool(Wiki, database=wiki_db))
    return tools


def run_faithfulness_experiment(api_key, prompts, expected_tools, sampling_ratio=0.5):
    """
    For each prompt:
    1. Run AgentSHAP to get SHAP values
    2. Remove the TOP SHAP tool and measure quality drop
    3. Remove the BOTTOM SHAP tool and measure quality drop

    Hypothesis: Removing high-SHAP tool should hurt more than removing low-SHAP tool
    """
    model = OpenAIModel(model_name="gpt-4o-mini", api_key=api_key)
    vectorizer = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    all_tools = create_tools()

    results = []

    for idx, (prompt, expected_tool) in enumerate(zip(prompts, expected_tools)):
        print(f"\n{'='*70}")
        print(f"FAITHFULNESS TEST {idx+1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"Expected tool: {expected_tool}")
        print(f"{'='*70}")

        # Step 1: Get SHAP values with all tools
        agent_shap = AgentSHAP(model=model, tools=all_tools, vectorizer=vectorizer, max_iterations=3)
        _, shap_values = agent_shap.analyze(prompt, sampling_ratio=sampling_ratio)

        baseline_response = agent_shap._baseline_response
        baseline_similarity = 1.0  # Reference point

        # Sort tools by SHAP value
        sorted_tools = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        top_tool = sorted_tools[0][0]
        top_shap = sorted_tools[0][1]
        bottom_tool = sorted_tools[-1][0]
        bottom_shap = sorted_tools[-1][1]

        print(f"\nSHAP Values: {shap_values}")
        print(f"Top tool: {top_tool} (SHAP={top_shap:.3f})")
        print(f"Bottom tool: {bottom_tool} (SHAP={bottom_shap:.3f})")

        # Step 2: Remove TOP SHAP tool and measure quality
        tools_without_top = [t for t in all_tools if t.name != top_tool]
        agent_no_top = AgentSHAP(model=model, tools=tools_without_top, vectorizer=vectorizer, max_iterations=3)
        response_no_top, _ = agent_no_top.analyze(prompt, sampling_ratio=0.3)

        # Calculate similarity to baseline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf = TfidfVectorizer()
        try:
            vecs = tfidf.fit_transform([baseline_response, agent_no_top._baseline_response])
            sim_without_top = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        except:
            sim_without_top = 0.0

        print(f"Similarity without {top_tool}: {sim_without_top:.3f}")

        # Step 3: Remove BOTTOM SHAP tool and measure quality
        tools_without_bottom = [t for t in all_tools if t.name != bottom_tool]
        agent_no_bottom = AgentSHAP(model=model, tools=tools_without_bottom, vectorizer=vectorizer, max_iterations=3)
        response_no_bottom, _ = agent_no_bottom.analyze(prompt, sampling_ratio=0.3)

        try:
            vecs = tfidf.fit_transform([baseline_response, agent_no_bottom._baseline_response])
            sim_without_bottom = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        except:
            sim_without_bottom = 0.0

        print(f"Similarity without {bottom_tool}: {sim_without_bottom:.3f}")

        # Record results
        results.append({
            "prompt": prompt,
            "expected_tool": expected_tool,
            "top_tool": top_tool,
            "top_shap": top_shap,
            "bottom_tool": bottom_tool,
            "bottom_shap": bottom_shap,
            "baseline_response": baseline_response[:200],
            "sim_without_top": sim_without_top,
            "sim_without_bottom": sim_without_bottom,
            "quality_drop_top": 1.0 - sim_without_top,
            "quality_drop_bottom": 1.0 - sim_without_bottom,
            "faithfulness_valid": (1.0 - sim_without_top) >= (1.0 - sim_without_bottom)
        })

        print(f"\nQuality drop removing {top_tool}: {1.0 - sim_without_top:.3f}")
        print(f"Quality drop removing {bottom_tool}: {1.0 - sim_without_bottom:.3f}")
        print(f"Faithfulness valid: {results[-1]['faithfulness_valid']}")

    return results


def plot_faithfulness_results(results, output_path):
    """Create visualization for faithfulness experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Quality drop comparison (bar chart)
    ax1 = axes[0]
    prompts_labels = [f"P{i+1}" for i in range(len(results))]
    x = np.arange(len(prompts_labels))
    width = 0.35

    drops_top = [r["quality_drop_top"] for r in results]
    drops_bottom = [r["quality_drop_bottom"] for r in results]

    bars1 = ax1.bar(x - width/2, drops_top, width, label='Remove Top-SHAP Tool', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, drops_bottom, width, label='Remove Bottom-SHAP Tool', color='blue', alpha=0.7)

    ax1.set_xlabel('Prompt')
    ax1.set_ylabel('Quality Drop (1 - similarity)')
    ax1.set_title('Quality Drop When Removing Tools')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts_labels)
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # Plot 2: Faithfulness validity (pie chart)
    ax2 = axes[1]
    valid_count = sum(1 for r in results if r["faithfulness_valid"])
    invalid_count = len(results) - valid_count

    ax2.pie([valid_count, invalid_count], labels=['Valid', 'Invalid'],
            colors=['green', 'red'], autopct='%1.0f%%', startangle=90)
    ax2.set_title(f'Faithfulness Validity\n({valid_count}/{len(results)} prompts)')

    # Plot 3: Scatter plot SHAP vs Quality Drop
    ax3 = axes[2]
    all_shaps = [r["top_shap"] for r in results] + [r["bottom_shap"] for r in results]
    all_drops = [r["quality_drop_top"] for r in results] + [r["quality_drop_bottom"] for r in results]
    colors = ['red'] * len(results) + ['blue'] * len(results)

    ax3.scatter(all_shaps, all_drops, c=colors, alpha=0.7, s=100)
    ax3.set_xlabel('SHAP Value')
    ax3.set_ylabel('Quality Drop')
    ax3.set_title('SHAP Value vs Quality Drop\n(Red=Top, Blue=Bottom)')

    # Add trend line
    z = np.polyfit(all_shaps, all_drops, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_shaps), max(all_shaps), 100)
    ax3.plot(x_line, p(x_line), "k--", alpha=0.5, label=f'Trend')
    ax3.legend()

    plt.suptitle('Experiment 2: Faithfulness Test\n(API-Bank Benchmark)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_results_to_csv(results, output_dir):
    """Save results to CSV."""
    df = pd.DataFrame(results)
    csv_path = output_dir / "exp2_faithfulness_results.csv"
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
    print(f"FAITHFULNESS EXPERIMENT")
    print(f"{'='*70}")
    print(f"Prompts: {len(prompts)}")
    for i, (p, e) in enumerate(zip(prompts, expected_tools)):
        print(f"  {i+1}. [{e}] {p[:50]}...")

    # Run experiment
    results = run_faithfulness_experiment(
        api_key=api_key,
        prompts=prompts,
        expected_tools=expected_tools,
        sampling_ratio=0.5
    )

    # Print summary
    print("\n" + "="*70)
    print("FAITHFULNESS SUMMARY")
    print("="*70)

    valid_count = sum(1 for r in results if r["faithfulness_valid"])
    print(f"\nFaithfulness Validity: {valid_count}/{len(results)} ({100*valid_count/len(results):.0f}%)")
    print(f"\nMean quality drop (remove top tool): {np.mean([r['quality_drop_top'] for r in results]):.3f}")
    print(f"Mean quality drop (remove bottom tool): {np.mean([r['quality_drop_bottom'] for r in results]):.3f}")

    # Save results
    save_results_to_csv(results, results_dir)

    # Plot
    output_path = results_dir / "exp2_faithfulness.png"
    plot_faithfulness_results(results, output_path)

    print(f"\n✓ Results saved to: {results_dir}/")
    print("  - exp2_faithfulness.png")
    print("  - exp2_faithfulness_results.csv")

    os.system(f"open {output_path}")
