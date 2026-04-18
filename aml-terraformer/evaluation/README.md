# Transaction Network Power Law Analysis

This directory contains scripts for evaluating transaction networks using power law distribution analysis.

## Files

- `powerlaw.py` - Core power law analysis module
- `analyze_transaction_powerlaw.py` - Main script to analyze transaction data
- `requirements.txt` - Python dependencies

## Usage

### Basic Usage

Run the analysis on the default dataset:

```bash
cd evaluation
export AML_DATA_DIR=../data
python analyze_transaction_powerlaw.py
```

This will:
1. Load transaction data from `$AML_DATA_DIR/transactions.csv` (default: `data/transactions.csv`)
2. Build a directed graph (nodes = accounts, edges = transactions)
3. Analyze in-degree, out-degree, and total degree distributions
4. Generate plots and statistical results in `./results/` directory

### Output

The script generates the following outputs in the `results/` directory:

- `in_degree_powerlaw.png` - In-degree distribution plot with power law fit
- `out_degree_powerlaw.png` - Out-degree distribution plot with power law fit
- `total_degree_powerlaw.png` - Total degree distribution plot with power law fit
- `powerlaw_analysis_results.txt` - Summary of statistical results

### Customization

To analyze a different CSV file or change output directory, modify the configuration in the `main()` function:

```python
csv_path = '../data/YOUR_FILE.csv'
output_dir = './your_output_dir'
```

## Graph Construction

The script constructs a directed graph where:
- **Nodes**: Bank accounts (identified by "Bank_Account" combination)
- **Edges**: Transactions between accounts
- **Edge weights**: Transaction amounts (summed if multiple transactions between same accounts)

## Power Law Analysis

The analysis fits the degree distribution to a power law and compares it with alternative distributions:
- Lognormal
- Exponential
- Truncated power law
- Stretched exponential

Results include:
- Power law exponent (alpha)
- Minimum degree threshold (xmin)
- Goodness of fit (KS statistic)
- Likelihood ratio tests comparing distributions

## Requirements

See `requirements.txt` for required Python packages.
