
import pandas as pd
from urllib.parse import urlparse

df = pd.read_excel(
    "data/compphish/Mapping_File.xlsx"
)

df = df.drop_duplicates(subset=["url"]).copy()


def inspect_url(url):
    try:
        parsed = urlparse(str(url).strip())

        return pd.Series({
            "valid": (
                parsed.scheme in ["http", "https"]
                and bool(parsed.hostname)
            ),
            "root_only": parsed.path in ["", "/"],
            "has_deep_path": parsed.path not in ["", "/"],
            "has_query": bool(parsed.query),
            "url_length": len(str(url))
        })
    except ValueError:
        return pd.Series({
            "valid": False,
            "root_only": False,
            "has_deep_path": False,
            "has_query": False,
            "url_length": len(str(url))
        })


print("Inspecting URL structure...")

features = df["url"].apply(inspect_url)

audit = pd.concat(
    [df.reset_index(drop=True),
     features.reset_index(drop=True)],
    axis=1
)

print("\nTOTAL UNIQUE URLS:", len(audit))
print("INVALID URLS:", (~audit["valid"]).sum())

print("\nROOT-ONLY URLS (%)")
print(
    audit.groupby("label")["root_only"]
    .mean().mul(100).round(2)
)

print("\nDEEP-PATH URLS (%)")
print(
    audit.groupby("label")["has_deep_path"]
    .mean().mul(100).round(2)
)

print("\nURLS WITH QUERY PARAMETERS (%)")
print(
    audit.groupby("label")["has_query"]
    .mean().mul(100).round(2)
)

print("\nURL LENGTH QUANTILES")
print(
    audit.groupby("label")["url_length"]
    .quantile([0.5, 0.9, 0.99])
)

print("\nLONGEST URL LENGTH BY CLASS")
print(
    audit.groupby("label")["url_length"].max()
)
