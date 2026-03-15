# Temporal Coherence Experiment

Experimental validation for the paper: *"The Temporal Coherence Problem: Synthetic Point-in-Time Environments for Evaluating LLM Agents with Dynamic Tool Dependencies"*

## What This Tests

A simulated Incident Root-Cause Analysis (RCA) agent diagnoses production incidents using 4 tools (metrics, alerts, deployments, log analysis). We test the agent under 6 conditions to show that temporal incoherence in tool data degrades diagnostic accuracy.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # Your OpenAI key
```

## Run

```bash
# Default: OpenAI gpt-4o-mini, 3 repeats
python run_experiment.py --verbose

# Use a different model
python run_experiment.py --model gpt-4o --verbose

# Use Anthropic Claude
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python run_experiment.py --provider anthropic --verbose

# More repeats for statistical confidence
python run_experiment.py --repeats 5 --verbose
```

## Cost Estimate

- 5 scenarios × 6 conditions × 3 repeats = 90 API calls
- Each call: ~2K input tokens, ~200 output tokens
- With gpt-4o-mini: approximately $0.05 total
- With gpt-4o: approximately $0.50 total

## Experimental Conditions

| Condition | What Changes | Expected Effect |
|-----------|-------------|-----------------|
| C1: Coherent Snapshot | All tools at same timestamp | Baseline — high accuracy |
| C2: Temporal Incoherence | Tools return data from different times | Agent gets contradictory signals |
| C3: Schema Drift | Alert service uses new format | Agent may misinterpret alerts |
| C4: Sub-Agent Reasoning Drift | Log analysis frames errors differently | Agent may reach different diagnosis |
| C5: Data Gap | New canary tool, data missing for 3/5 scenarios | Agent can't use new tool fully |
| C6: Restored Snapshot | Synthetic coherent snapshot applied | Accuracy should recover to ~C1 |

## Scenarios

1. **Bad Config Deployment** — Misconfigured timeout in payment-service
2. **Database Connection Pool Exhaustion** — Traffic surge saturates connection pool
3. **Upstream Dependency Timeout** — Third-party payment gateway degradation
4. **Memory Leak** — Code regression causing gradual heap exhaustion
5. **DNS Resolution Failure** — Infrastructure DNS config change breaks service discovery

## Output

Results are printed as a summary table and saved to `results.json` with full per-run details.

## Project Structure

```
temporal-coherence-experiment/
├── README.md
├── requirements.txt
├── run_experiment.py          # Main experiment runner
└── scenarios/
    ├── scenario_1_bad_deploy.json
    ├── scenario_2_db_pool.json
    ├── scenario_3_upstream_timeout.json
    ├── scenario_4_memory_leak.json
    └── scenario_5_dns_failure.json
```
