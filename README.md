# TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering

TransXion is a benchmark ecosystem for AML research that addresses two fundamental limitations of existing datasets: sparse node-level semantics and template-driven anomaly injection. By jointly modeling persistent entity profiles, profile-conditioned normal behavior, and adaptive non-template anomaly synthesis, TransXion captures realistic interactions between transactional topology and semantic context that are central to operational AML.

## The Problem with Existing AML Benchmarks

| Limitation | Effect |
|---|---|
| Edge-centric logs, no entity context | Detectors rely solely on structural motifs; node-level semantic reasoning is untestable |
| Template-driven anomaly injection | Benchmarks encode fixed patterns, enabling shortcut learning rather than genuine detection |

TransXion addresses both by construction: normal transactions are grounded in rich demographic and behavioral profiles, and illicit subgraphs are synthesized via budget-constrained adversarial graph edits rather than hardcoded templates.

---

## Three-Step Pipeline

```
Step 1 ── txsim ──────────────────────────────────────────────────────────────
         Profile-aware normal backbone: generate entity profiles and
         time-ordered transactions respecting behavioral priors.
              │
              ▼
Step 2 ── aml-terraformer ────────────────────────────────────────────────────
         Anomaly synthesis: derive illicit clusters from real laundering
         subgraphs via four budget-constrained graph edit operations.
              │
              ▼
Step 3 ── Profile-conditioned embedding ──────────────────────────────────────
         Insert each illicit cluster into the normal backbone by aligning
         anomalous roles to compatible entities and feasible time windows.
```

---

## Dataset

The released dataset spans one full year of simulated financial activity.

| Statistic | Value |
|---|---|
| Days | 365 |
| Unique accounts | 47,526 |
| Total transactions | 3,029,170 |
| Laundering transactions | 4,641 |
| Prevalence | ~0.15% (1 in 653) |

Each account carries structured profile attributes (demographics, behavioral tags, activity distributions). Each transaction records sender/receiver bank–account pairs, timestamp, payment amount in payment currency, received amount in receiving currency, payment format, and a binary `is_laundering` label.

| File | Size | Description |
|---|---|---|
| `data/tx.csv` | 234 MB | Full transaction dataset (tracked via Git LFS) |
| `data/person.csv` | 3.1 MB | Synthetic person account profiles |
| `data/merchant.csv` | 196 KB | Synthetic merchant account profiles |

`data/tx.csv` is stored via Git LFS. Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/chaos-max/TransXion.git
```

---

## Repository Structure

```
TransXion/
├── data/
│   ├── tx.csv                   # Full transaction dataset, 3M rows (Git LFS)
│   ├── person.csv               # Synthetic person account profiles
│   └── merchant.csv             # Synthetic merchant account profiles
│
├── txsim/                       # Step 1: Normal transaction backbone
│   ├── LLMGraph/
│   │   ├── agent/               # LLM planning and transaction agents
│   │   ├── environments/        # Time-stepping simulation environment
│   │   ├── manager/             # Account and transaction state management
│   │   ├── prompt/              # Prompt templates (daily scenario, geo planning)
│   │   ├── tasks/transaction/   # config.yaml and seed data (profiles, merchants)
│   │   ├── llms/                # Model config (default_model_configs.json)
│   │   └── utils/               # Sampling, I/O, parallel utilities
│   ├── main_txn_async.py        # Main entry point
│   ├── run_txn.sh               # Launch script
│   ├── scripts/                 # Setup verification scripts
│   └── requirements.txt
│
└── aml-terraformer/             # Steps 2–3: Anomaly synthesis and embedding
    ├── src/aml_terraformer/
    │   ├── agent/               # LLM agent (prompt, decision, sanitization)
    │   ├── core/                # Clustering, validation, preprocessing
    │   ├── tools/               # Graph edit operations (inject/merge/split/adjust)
    │   ├── monitor/             # Detection monitors (rule, GBT, GNN)
    │   ├── pipeline/            # Run orchestration, logging, reporting
    │   └── rl/                  # GRPO reinforcement learning trainer
    ├── rule/                    # AML rule engine and feature builder
    │   └── data/rule.json       # Rule definitions (S1, S3, S5, S6, S7, ...)
    ├── scripts/                 # Training and evaluation scripts
    ├── docs/                    # Detailed documentation
    └── pyproject.toml
```

---

## Step 1: txsim — Normal Transaction Backbone

`txsim` generates large-scale, statistically consistent synthetic transaction data using a hierarchical LLM-guided simulation. The design separates "planning" from "execution": an LLM provides distributional guidance at macro and meso scales, while all per-transaction field generation is handled by deterministic rules to guarantee constraint consistency and reproducibility.

### Architecture

- **Macro level** — An LLM produces daily volume multipliers and scenario themes (e.g., holiday surges, promotional activity). This runs once per simulated day and shapes the overall transaction density and regional intensity.
- **Meso level** — Per-geography transaction planning: cross-border ratios, payment format distributions, amount buckets. Runs once per time window per geographic region.
- **Micro level** — Rule-driven field sampling: sender/receiver selection via preference memory and hotspot reinforcement, timestamps, amounts, currencies, fees.

Key properties:
- Async parallel generation across 20 geographic regions
- Counterparty preference memory with decay models realistic repeat-transaction behavior
- Checkpoint/resume support for long-running generation jobs
- Pluggable LLM backend: local vLLM or API (OpenAI, DeepSeek)

### Installation

```bash
cd txsim
pip install -r requirements.txt
```

AgentScope (the multi-agent framework) is installed separately as it is excluded from version control:

```bash
pip install agentscope
```

### Configuration

The default configuration uses the DeepSeek API. Set your API key in `LLMGraph/llms/default_model_configs.json`:

```json
{
  "config_name": "deepseek",
  "model_type": "openai_chat",
  "model_name": "deepseek-reasoner",
  "api_key": "YOUR_DEEPSEEK_API_KEY",
  "client_args": { "base_url": "https://api.deepseek.com" }
}
```

To use a local vLLM server instead, change all role mappings in `LLMGraph/tasks/transaction/config.yaml` from `"deepseek"` to `"vllm_local"`, then start the local service:

```bash
bash manage_llm_service.sh start   # starts vLLM on localhost:8001
```

Simulation parameters (duration, target transaction count, number of regions, behavioral coefficients) are set in `LLMGraph/tasks/transaction/config.yaml`.

### Running

```bash
bash run_txn.sh start
# Output: transaction_output/transactions_YYYY-MM-DD.csv
```

For a quick environment check before running:

```bash
python scripts/check_txsim_setup.py
```

### Output Format

Transactions are written to date-partitioned CSV files:

```
transaction_output/
├── transactions_2025-01-01.csv
├── transactions_2025-01-02.csv
...
```

Each row:

| Column | Description |
|---|---|
| `Timestamp` | Transaction datetime |
| `From Bank` | Sender bank ID |
| `Account` (from) | Sender account number |
| `To Bank` | Receiver bank ID |
| `Account` (to) | Receiver account number |
| `Amount Paid` | Payment amount |
| `Payment Currency` | Currency of payment |
| `Amount Received` | Received amount after fees/FX |
| `Receiving Currency` | Currency received |
| `Payment Format` | Mobile / Card / Transfer / Cash |

---

## Steps 2–3: aml-terraformer — Anomalous Transaction Synthesis

`aml-terraformer` takes labeled transaction data containing real money-laundering clusters and uses an LLM agent to apply budget-constrained graph edits, producing illicit subgraphs that are behaviorally plausible and structurally non-trivial for detectors. The resulting clusters are then embedded into the normal backbone with profile-conditioned alignment.

### Four Graph Edit Operations

| Operation | Description |
|---|---|
| **Intermediary Injection** | Insert a pass-through account into a transaction chain to break direct linkage |
| **Account Merging** | Consolidate multiple accounts into one to obscure coordination signals |
| **Account Splitting** | Fragment a high-risk account across two new accounts to dilute risk signals |
| **Transaction Adjustment** | Shift timestamps or amounts to break statistical detection patterns |

The LLM agent selects actions from scored candidates under an edit budget. All operations include strict validation with automatic rollback on failure.

### Installation

```bash
cd aml-terraformer
pip install -e .

# All LLM providers:
pip install -e ".[all]"
# Or for a specific provider:
pip install -e ".[openai]"
pip install -e ".[deepseek]"
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` (auto-loaded at startup via `python-dotenv`):

```bash
AML_DATA_DIR=data                        # directory containing accounts.csv and transactions.csv
DEEPSEEK_API_KEY=your-api-key            # or OPENAI_API_KEY
QWEN_MODEL_PATH=/path/to/Qwen2.5-7B     # for local model inference
MULTIGNN_PATH=/path/to/multignn          # optional, for GNN-based monitoring
```

### Input Data Format

**accounts.csv**

| Column | Required |
|---|---|
| `Bank Name` | Yes |
| `Bank ID` | Yes |
| `Account Number` | Yes |
| `Entity ID` | No |
| `Entity Name` | No |

**transactions.csv**

```
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
```

The two `Account` columns are From Account and To Account respectively. `Is Laundering`: 1 = laundering, 0 = normal.

### Running Perturbation

```bash
# DeepSeek API
python scripts/run_perturbation.py \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --output output/exp_1 \
    --provider deepseek \
    --api-key $DEEPSEEK_API_KEY \
    --max-cluster-size 100

# Local model
python scripts/run_perturbation.py \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --output output/exp_1 \
    --provider local \
    --model $QWEN_MODEL_PATH \
    --device cuda
```

Output:

```
output/exp_1/
├── transactions_perturbed.csv      # perturbed transactions
├── accounts_perturbed.csv          # updated account list
├── perturb_log.jsonl               # per-operation audit log
├── summary_report.json             # aggregate statistics
└── cluster_*_comparison.png        # before/after visualization per cluster
```

### (Optional) Fine-tune with GRPO

GRPO (Group Relative Policy Optimization) fine-tunes a local LLM to learn a better perturbation policy guided by monitor feedback.

```bash
# Data collection pass (uses API model, no gradient updates)
python scripts/train_grpo.py \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --output output/grpo_train \
    --provider deepseek \
    --api-key $DEEPSEEK_API_KEY \
    --train-mode collect

# Training pass (uses local model, applies LoRA fine-tuning)
python scripts/train_grpo.py \
    --accounts $AML_DATA_DIR/accounts.csv \
    --transactions $AML_DATA_DIR/transactions.csv \
    --output output/grpo_train \
    --provider local \
    --model $QWEN_MODEL_PATH \
    --train-mode train \
    --use-lora \
    --device cuda
```

Reward weights: `--w-detection` (default 10.0), `--w-validity` (1.0), `--w-budget` (0.1), `--w-final-bonus` (20.0).

### AML Rule Engine

A standalone rule-based AML engine is provided in `rule/` for evaluation and feature engineering:

```bash
# Build rule features from transactions
python rule/build_rule_features.py --input transactions.csv --output features.csv

# Run rule engine (outputs hit flags per transaction)
python rule/run_rule_engine.py --features features.csv --rules rule/data/rule.json

# Analyze rule hit distribution
bash scripts/run_rule_analysis.sh
```

---

## Evaluation

TransXion yields the lowest Average Precision and F1 across all GNN detector families, confirming it provides a substantially more demanding evaluation setting than existing benchmarks.

**GNN detection performance (lower AP/F1 = harder benchmark):**

| Dataset | AUC | AP | F1 |
|---|---|---|---|
| AMLSim | 0.930–0.949 | 0.458–0.729 | 0.486–0.803 |
| AMLWorld | 0.978–0.986 | 0.395–0.638 | 0.452–0.710 |
| SAML-D | 0.999–1.000 | 0.971–0.995 | 0.913–0.968 |
| **TransXion** | **0.952–0.969** | **0.269–0.351** | **0.425–0.487** |

Detection gains are consistently larger when node profile features are included (+29.5% ΔAUC for LightGBM), confirming that entity-level context provides signal complementary to transaction-level topology.

**Ablation:** RL-Optimized synthesis (GRPO-trained policy) produces the hardest benchmark configuration, with lower AP and F1 than LLM-only or random baselines across both GNN and gradient-boosted detectors.

---

## Requirements

- Python 3.10+
- Git LFS (for `data/tx.csv`)
- LLM backend: local [vLLM](https://github.com/vllm-project/vllm) server, or API access to OpenAI / DeepSeek
- GPU recommended for local model inference (8 GB+ VRAM)

---

## License

See [aml-terraformer/LICENSE](./aml-terraformer/LICENSE) and [txsim/LICENSE](./txsim/LICENSE).