"""Quick status check — how many runs are done."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path
import pandas as pd

p = Path("outputs/metrics/results.csv")
if p.exists():
    df = pd.read_csv(p)
    print(f"Total completed rows: {len(df)}")
    print("\nBreakdown by dataset / model / noise_type:")
    print(df.groupby(["dataset", "model", "noise_type"]).size().to_string())
    print(f"\nDatasets seen: {sorted(df['dataset'].unique())}")
    print(f"Models seen:   {sorted(df['model'].unique())}")
else:
    print("No results.csv yet — no runs completed.")
