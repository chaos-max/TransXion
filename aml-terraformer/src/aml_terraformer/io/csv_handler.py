"""CSV reading and writing with special handling for duplicate column names."""

import pandas as pd
from pathlib import Path


def read_accounts(path: str) -> pd.DataFrame:
    """Read accounts CSV.

    Expected columns: Bank Name, Bank ID, Account Number, Entity ID, Entity Name
    """
    df = pd.read_csv(path)
    required_cols = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in accounts: {col}")
    return df


def read_transactions(path: str) -> pd.DataFrame:
    """Read transactions CSV with two 'Account' columns.

    Original columns: Timestamp, From Bank, Account, To Bank, Account, Amount Received,
                      Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering

    Internally rename the two 'Account' columns to 'From Account' and 'To Account'.
    """
    # Read with header=None to get raw columns
    with open(path, 'r') as f:
        header_line = f.readline().strip()

    # Parse header manually
    cols = header_line.split(',')

    # Expected column order with two 'Account' columns at positions 2 and 4
    expected_order = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]

    # Rename the two 'Account' columns
    new_cols = []
    account_count = 0
    for col in cols:
        col = col.strip()
        if col == "Account":
            if account_count == 0:
                new_cols.append("From Account")
                account_count += 1
            elif account_count == 1:
                new_cols.append("To Account")
                account_count += 1
            else:
                new_cols.append(col)
        else:
            new_cols.append(col)

    # Read CSV with renamed columns
    df = pd.read_csv(path, skiprows=1, names=new_cols)

    # Verify all expected columns exist
    required_cols = [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in transactions: {col}")

    return df


def write_accounts(df: pd.DataFrame, path: str):
    """Write accounts CSV.

    May contain additional columns like Status, MergedInto beyond the original 5.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_transactions(df: pd.DataFrame, path: str):
    """Write transactions CSV with two 'Account' columns restored.

    Internal columns 'From Account' and 'To Account' are renamed back to 'Account'.
    Column order must match original:
    Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,
    Amount Paid,Payment Currency,Payment Format,Is Laundering
    """
    # Make a copy to avoid modifying original
    df_out = df.copy()

    # Ensure correct column order
    output_order = [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]

    # Reorder columns
    df_out = df_out[output_order]

    # Write to CSV with custom header
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Create header with two 'Account' columns
    header = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]

    # Write manually
    with open(path, 'w') as f:
        # Write header
        f.write(','.join(header) + '\n')
        # Write data
        df_out.to_csv(f, index=False, header=False)
