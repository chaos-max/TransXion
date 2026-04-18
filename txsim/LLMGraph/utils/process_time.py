
from datetime import datetime, timedelta
import pandas as pd


def transfer_time(timestamp):
    if isinstance(timestamp,str):
        try:
            time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()
        except ValueError:
            try:
                time = datetime.strptime(timestamp, '%Y-%m-%d').date()
            except ValueError:
                try:
                    time = datetime.strptime(timestamp, '%Y-%m').date()
                except ValueError:
                    raise ValueError(f"timestamp {timestamp} not available")
    elif isinstance(timestamp,int):
        time = datetime.fromtimestamp(timestamp).date()
    elif isinstance(timestamp,pd.Timestamp):
        time = timestamp.date()
    elif isinstance(timestamp,datetime):
        time = timestamp.date()
    else:
        time = timestamp
    return time