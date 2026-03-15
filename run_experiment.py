"""
Temporal Coherence Experiment for LLM Agent Evaluation
======================================================
Validates that temporal incoherence, tool drift, and sub-agent reasoning drift
degrade incident RCA agent performance, and that synthetic snapshots restore it.

Usage:
    pip install openai tabulate
    export OPENAI_API_KEY="sk-..."
    python run_experiment.py

    # Or use Anthropic:
    pip install anthropic tabulate
    export ANTHROPIC_API_KEY="sk-ant-..."
    python run_experiment.py --provider anthropic
"""

import json
import os
import sys
import argparse
import copy
from pathlib import Path
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
NUM_REPEATS = 3  # Repeat each condition for consistency measurement

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer performing incident root cause analysis.
You will be given data from four monitoring tools about a production incident:
1. METRICS: Service latency, error rates, and request counts over time
2. ALERTS: Triggered alerts with severity and timestamps
3. DEPLOYMENTS: Recent deployments with timestamps and change descriptions
4. LOG ANALYSIS: Summarized error log patterns from a log analysis system

Your job is to:
1. Analyze all the data together
2. Identify the most likely ROOT CAUSE of the incident
3. Cite specific evidence supporting your diagnosis

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
    "root_cause": "<one of: bad_deployment, database_connection_pool, upstream_dependency, memory_leak, dns_failure, config_change, network_issue, unknown>",
    "confidence": <0.0 to 1.0>,
    "summary": "<one sentence explanation>",
    "evidence": ["<evidence point 1>", "<evidence point 2>", "<evidence point 3>"]
}"""

# ---------------------------------------------------------------------------
# LLM Client Abstraction
# ---------------------------------------------------------------------------

def call_llm(prompt: str, provider: str = "openai", model: str = None) -> str:
    """Call LLM API and return response text."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = model or "gpt-4o-mini"  # Cheap, fast, good enough
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        model = model or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Scenario Loading
# ---------------------------------------------------------------------------

def load_scenarios() -> list:
    """Load all scenario JSON files."""
    scenarios = []
    for f in sorted(SCENARIOS_DIR.glob("scenario_*.json")):
        with open(f) as fh:
            scenarios.append(json.load(fh))
    return scenarios


# ---------------------------------------------------------------------------
# Condition Builders
# ---------------------------------------------------------------------------
# Each function takes a scenario and returns the tool data dict the agent sees.

def build_agent_prompt(metrics, alerts, deployments, log_analysis, canary=None) -> str:
    """Build the agent prompt from tool responses."""
    sections = [
        "=== METRICS DATA ===",
        json.dumps(metrics, indent=2),
        "\n=== ALERTS DATA ===",
        json.dumps(alerts, indent=2),
        "\n=== DEPLOYMENT HISTORY ===",
        json.dumps(deployments, indent=2),
        "\n=== LOG ANALYSIS (from log analysis sub-agent) ===",
        json.dumps(log_analysis, indent=2),
    ]
    if canary:
        sections.extend([
            "\n=== CANARY METRICS (from canary analysis tool) ===",
            json.dumps(canary, indent=2),
        ])
    sections.append(
        "\n\nBased on ALL the data above, diagnose the root cause of this incident."
    )
    return "\n".join(sections)


def condition_c1_coherent(scenario: dict) -> str:
    """C1: Coherent snapshot - all tools at same point in time."""
    s = scenario["coherent_snapshot"]
    return build_agent_prompt(
        s["metrics"], s["alerts"], s["deployments"], s["log_analysis"]
    )


def condition_c2_temporal_incoherent(scenario: dict) -> str:
    """C2: Temporal incoherence - tools return data from different timestamps.
    - Metrics: correct (at incident time) - shows the real problem
    - Alerts: from a later time window - includes resolved alerts AND new unrelated alerts
    - Deployments: post-incident - shows hotfixes/rollbacks that look like they caused the issue
    - Logs: from a later time window - shows errors from post-incident activity, not the original cause
    The combination actively misleads: metrics say one thing, everything else points elsewhere.
    """
    s = scenario["coherent_snapshot"]
    d = scenario["drifted_data"]
    return build_agent_prompt(
        s["metrics"],                        # Correct: shows the incident
        d["alerts_misleading"],              # Drifted: resolved + new unrelated alerts
        d["deployments_misleading"],         # Drifted: post-incident deploys look causal
        d["logs_misleading"]                 # Drifted: logs from later show different errors
    )


def condition_c3_schema_drift(scenario: dict) -> str:
    """C3: Tool drift - alert service returns new schema format.
    Agent prompt expects {severity: high/medium/low} but gets {priority: P1/P2/P3}
    with different field names throughout.
    """
    s = scenario["coherent_snapshot"]
    d = scenario["drifted_data"]
    return build_agent_prompt(
        s["metrics"],
        d["alerts_new_schema"],               # Drifted: new schema
        s["deployments"],
        s["log_analysis"]
    )


def condition_c4_subagent_reasoning_drift(scenario: dict) -> str:
    """C4: Sub-agent reasoning drift - log analysis sub-agent categorizes
    errors differently (e.g., 'network' instead of 'application-level'),
    leading the orchestrator to a different diagnosis.
    """
    s = scenario["coherent_snapshot"]
    d = scenario["drifted_data"]
    return build_agent_prompt(
        s["metrics"],
        s["alerts"],
        s["deployments"],
        d["log_analysis_different_reasoning"]  # Different analytical framing
    )


def condition_c5_data_gap(scenario: dict) -> str:
    """C5: Forward-looking data gap - canary metrics tool added but
    data only available for some scenarios.
    """
    s = scenario["coherent_snapshot"]
    canary = scenario.get("canary_metrics", {})
    canary_data = canary.get("response") if canary.get("available") else {
        "status": "NO_DATA",
        "message": "Canary metrics not available for this deployment. Tool was not active during this incident."
    }
    return build_agent_prompt(
        s["metrics"], s["alerts"], s["deployments"], s["log_analysis"],
        canary=canary_data
    )


def condition_c6_restored(scenario: dict) -> str:
    """C6: Restored snapshot - synthetic point-in-time snapshot applied.
    This is equivalent to C1 (coherent) + canary data synthetically generated
    for all scenarios, demonstrating that restoration recovers baseline.
    """
    s = scenario["coherent_snapshot"]
    canary = scenario.get("canary_metrics", {})
    
    # For scenarios without real canary data, generate synthetic placeholder
    if canary.get("available"):
        canary_data = canary["response"]
    else:
        # Synthetic backfill: generate a "no anomaly in canary" signal
        # since there was no relevant canary deployment for this incident type
        canary_data = {
            "status": "SYNTHETIC_BACKFILL",
            "message": "No canary deployment associated with this incident. Canary analysis not applicable.",
            "verdict": "NOT_APPLICABLE"
        }
    
    return build_agent_prompt(
        s["metrics"], s["alerts"], s["deployments"], s["log_analysis"],
        canary=canary_data
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def parse_agent_response(response_text: str) -> dict:
    """Parse the agent's JSON response, handling potential formatting issues."""
    text = response_text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"root_cause": "parse_error", "confidence": 0, "summary": "Failed to parse", "evidence": []}


def evaluate_response(parsed: dict, ground_truth: dict) -> dict:
    """Evaluate a parsed agent response against ground truth."""
    correct_cause = parsed.get("root_cause", "").lower() == ground_truth["root_cause"].lower()
    
    # Evidence quality: what fraction of ground truth evidence points are mentioned
    gt_evidence = ground_truth.get("key_evidence", [])
    agent_evidence = parsed.get("evidence", [])
    agent_evidence_text = " ".join(agent_evidence).lower()
    
    evidence_hits = 0
    for gt_point in gt_evidence:
        # Check if key terms from this evidence point appear in agent's evidence
        key_terms = [t.lower() for t in gt_point.split() if len(t) > 4]
        matches = sum(1 for t in key_terms if t in agent_evidence_text)
        if matches >= len(key_terms) * 0.4:  # 40% term overlap threshold
            evidence_hits += 1
    
    evidence_quality = evidence_hits / len(gt_evidence) if gt_evidence else 0
    
    return {
        "correct": correct_cause,
        "predicted_cause": parsed.get("root_cause", "unknown"),
        "confidence": parsed.get("confidence", 0),
        "evidence_quality": round(evidence_quality, 2),
        "summary": parsed.get("summary", "")
    }


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

CONDITIONS = {
    "C1: Coherent Snapshot": condition_c1_coherent,
    "C2: Temporal Incoherence": condition_c2_temporal_incoherent,
    "C3: Schema Drift": condition_c3_schema_drift,
    "C4: Sub-Agent Reasoning Drift": condition_c4_subagent_reasoning_drift,
    "C5: Data Gap": condition_c5_data_gap,
    "C6: Restored Snapshot": condition_c6_restored,
}


def run_experiment(provider: str, model: str = None, repeats: int = NUM_REPEATS, verbose: bool = False):
    """Run the full experiment across all conditions and scenarios."""
    scenarios = load_scenarios()
    print(f"\nLoaded {len(scenarios)} scenarios")
    print(f"Provider: {provider}, Model: {model or 'default'}")
    print(f"Repeats per condition: {repeats}")
    print(f"Total LLM calls: {len(CONDITIONS) * len(scenarios) * repeats}")
    print("=" * 70)
    
    # Results storage
    all_results = {}
    
    for cond_name, cond_fn in CONDITIONS.items():
        print(f"\n--- {cond_name} ---")
        cond_results = []
        
        for scenario in scenarios:
            scenario_results = []
            prompt = cond_fn(scenario)
            
            for rep in range(repeats):
                try:
                    response_text = call_llm(prompt, provider=provider, model=model)
                    parsed = parse_agent_response(response_text)
                    evaluation = evaluate_response(parsed, scenario["ground_truth"])
                    scenario_results.append(evaluation)
                    
                    status = "✓" if evaluation["correct"] else "✗"
                    if verbose:
                        print(f"  {scenario['title']} (rep {rep+1}): {status} "
                              f"predicted={evaluation['predicted_cause']} "
                              f"evidence={evaluation['evidence_quality']}")
                    
                except Exception as e:
                    print(f"  ERROR on {scenario['title']} rep {rep+1}: {e}")
                    scenario_results.append({
                        "correct": False, "predicted_cause": "error",
                        "confidence": 0, "evidence_quality": 0, "summary": str(e)
                    })
            
            cond_results.append({
                "scenario": scenario["title"],
                "ground_truth": scenario["ground_truth"]["root_cause"],
                "runs": scenario_results
            })
        
        all_results[cond_name] = cond_results
    
    return all_results


def compute_summary(all_results: dict, num_scenarios: int, repeats: int) -> list:
    """Compute summary statistics for the results table."""
    summary_rows = []
    
    for cond_name, cond_results in all_results.items():
        total_correct = 0
        total_runs = 0
        evidence_scores = []
        consistency_scores = []
        
        for scenario_result in cond_results:
            runs = scenario_result["runs"]
            correct_in_runs = sum(1 for r in runs if r["correct"])
            total_correct += correct_in_runs
            total_runs += len(runs)
            evidence_scores.extend(r["evidence_quality"] for r in runs)
            
            # Consistency: did the agent give the same answer across repeats?
            predictions = [r["predicted_cause"] for r in runs]
            if predictions:
                most_common = max(set(predictions), key=predictions.count)
                consistency = predictions.count(most_common) / len(predictions)
                consistency_scores.append(consistency)
        
        # Per-scenario accuracy (majority vote across repeats)
        scenario_accuracy = 0
        for scenario_result in cond_results:
            runs = scenario_result["runs"]
            correct_count = sum(1 for r in runs if r["correct"])
            if correct_count > len(runs) / 2:
                scenario_accuracy += 1
        
        avg_evidence = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        summary_rows.append({
            "Condition": cond_name,
            "Accuracy (per-run)": f"{total_correct}/{total_runs} ({100*total_correct/total_runs:.0f}%)",
            "Accuracy (per-scenario)": f"{scenario_accuracy}/{num_scenarios}",
            "Avg Evidence Quality": f"{avg_evidence:.2f}",
            "Avg Consistency": f"{avg_consistency:.2f}",
        })
    
    return summary_rows


def print_detailed_results(all_results: dict):
    """Print per-scenario breakdown."""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS (per scenario)")
    print("=" * 70)
    
    for cond_name, cond_results in all_results.items():
        print(f"\n--- {cond_name} ---")
        for sr in cond_results:
            runs = sr["runs"]
            correct = sum(1 for r in runs if r["correct"])
            preds = [r["predicted_cause"] for r in runs]
            print(f"  {sr['scenario']:40s} | GT: {sr['ground_truth']:25s} | "
                  f"Correct: {correct}/{len(runs)} | Predictions: {preds}")


def save_results(all_results: dict, summary: list, output_path: str = "results.json"):
    """Save full results to JSON."""
    output = {
        "summary": summary,
        "detailed": {}
    }
    for cond_name, cond_results in all_results.items():
        output["detailed"][cond_name] = []
        for sr in cond_results:
            output["detailed"][cond_name].append({
                "scenario": sr["scenario"],
                "ground_truth": sr["ground_truth"],
                "runs": sr["runs"]
            })
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temporal Coherence Experiment")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                       help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                       help="Model name (default: gpt-4o-mini for openai, claude-sonnet-4-20250514 for anthropic)")
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS,
                       help=f"Repeats per condition (default: {NUM_REPEATS})")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-run details")
    parser.add_argument("--output", default="results.json",
                       help="Output file for results (default: results.json)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEMPORAL COHERENCE EXPERIMENT")
    print("Evaluating LLM Agent Performance Under Temporal Inconsistency")
    print("=" * 70)
    
    # Run experiment
    all_results = run_experiment(
        provider=args.provider,
        model=args.model,
        repeats=args.repeats,
        verbose=args.verbose
    )
    
    # Compute and display summary
    scenarios = load_scenarios()
    summary = compute_summary(all_results, len(scenarios), args.repeats)
    
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    print(tabulate(summary, headers="keys", tablefmt="grid"))
    
    # Print detailed breakdown
    print_detailed_results(all_results)
    
    # Save results
    save_results(all_results, summary, args.output)
    
    # Print interpretation guidance
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Expected patterns:
  C1 (Coherent):     High accuracy — baseline, all data consistent
  C2 (Temporal):     Lower accuracy — agent sees contradictory data from different times
  C3 (Schema):       Lower accuracy — agent may misinterpret changed alert format
  C4 (Sub-Agent):    Moderate drop — different log analysis framing may mislead
  C5 (Data Gap):     Similar to C1 for scenarios with canary data, 
                     but agent can't leverage canary tool for 3/5 scenarios
  C6 (Restored):     Accuracy recovers to near-C1 levels — validates snapshot approach

Key metric for the paper:
  - Accuracy gap between C1 and C2 demonstrates the temporal coherence problem
  - Accuracy recovery in C6 validates the proposed design patterns
  - C4 vs C1 gap shows sub-agent reasoning drift is a distinct challenge
""")


if __name__ == "__main__":
    main()
