# TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering

[![KDD 2026](https://img.shields.io/badge/KDD-2026-blue.svg)](https://doi.org/10.1145/3770855.3817550)
[![arXiv](https://img.shields.io/badge/arXiv-2604.17420-b31b1b.svg)](https://arxiv.org/abs/2604.17420)
[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-orange.svg)](https://doi.org/10.1145/3770855.3817550)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/chaos-max/TransXion)

## 💡 Updates

- 🎉 **TransXion has been accepted at KDD 2026 (Datasets and Benchmarks Track)!** 🚀
- 📄 Our paper is now available on [ACM Digital Library](https://doi.org/10.1145/3770855.3817550) and [arXiv](https://arxiv.org/abs/2604.17420).

## 📖 Overview
![TransXion Framework](pipeline.pdf)

**TransXion** is a realistic graph benchmark for **Anti-Money Laundering (AML)** research. It combines profile-aware transaction simulation with adaptive, non-template anomaly generation, going beyond existing benchmarks that lack rich entity information or rely on fixed laundering patterns.

## 🎯 Key Features

- **👥 Profile-Aware Entities:** Rich demographic and behavioral profiles for realistic account activity.
- **💸 Realistic Transactions:** Time-ordered financial interactions guided by entity behavior.
- **🧩 Adaptive Anomalies:** Flexible illicit subgraph generation with graph-edit operations instead of fixed templates.
- **🤖 LLM + RL Optimization:** LLM-guided anomaly synthesis with optional GRPO training.
- **🔍 Challenging AML Evaluation:** A benchmark designed to test context-aware and robust detection models.

## ⚙️ Framework

TransXion follows a simple three-stage pipeline:

1. **🏦 Normal Simulation (`txsim`):** Generate profile-aware entities and everyday financial transactions.
2. **🛠️ Anomaly Synthesis (`aml-terraformer`):** Create diverse illicit transaction patterns with constrained graph edits.
3. **🔗 Profile-Conditioned Embedding:** Integrate anomalous subgraphs into a realistic transaction network.

## 📊 Dataset

The released dataset covers **365 days** of simulated financial activity:

| Statistic | Value |
| --- | ---: |
| 👤 Accounts | 47,526 |
| 💳 Transactions | 3,029,170 |
| 🚨 Laundering transactions | 4,641 |
| 📉 Laundering prevalence | ~0.15% |

**Data files:** `data/tx.csv`, `data/person.csv`, and `data/merchant.csv`. The transaction file uses **Git LFS**.

## 🚀 Quick Start

### Download the dataset

```bash
git lfs install
git clone https://github.com/chaos-max/TransXion.git
cd TransXion
```

### Run transaction simulation

```bash
cd txsim
pip install -r requirements.txt
pip install agentscope
# Configure your LLM provider in LLMGraph/llms/default_model_configs.json
bash run_txn.sh start
```

For anomaly synthesis and optional GRPO training, see [`aml-terraformer/`](./aml-terraformer/). 🔧

## 📈 Evaluation

TransXion provides a more challenging benchmark for AML detectors. Across evaluated GNN models, it produces lower **Average Precision (AP)** and **F1** than AMLSim, AMLWorld, and SAML-D, highlighting the importance of realistic anomalies and entity-level context.

## 🧾 Citation

If you find TransXion useful, please consider citing our KDD 2026 paper! ⭐

```bibtex
@inproceedings{chen2026transxion,
  title={Transxion: A high-fidelity graph benchmark for realistic anti-money laundering},
  author={Chen, Keyang and Jiang, Mingxuan and Zhao, Yongsheng and Li, Zeping and Chen, Zaiyuan and Luo, Weiqi and Li, Zhixin and Liu, Sen and Jing, Yinan and Ye, Guangnan and others},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={8707--8718},
  year={2026},
  doi= {10.1145/3770855.3817550}
}
```

## 📄 License

See [`txsim/LICENSE`](./txsim/LICENSE) and [`aml-terraformer/LICENSE`](./aml-terraformer/LICENSE) for license details.

---

⭐ **If you find this project useful, consider giving it a star!**
