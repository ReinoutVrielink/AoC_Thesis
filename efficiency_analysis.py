import json
import re
import numpy as np
import pandas as pd
from clustering import get_feature_columns

# this file contains the code for conducting the efficienicy analysis from RQ3
# it is based on the code from Martijn Huls, who provided me with the calculated runtimes per solution
# the runtimes were measured on different hardware/inputs than ours, so they shouldn't be taken as a controlled benchmark

def normalize_code(code):
    normalizedcode = code.strip()
    # Replace only the filename inside open(...) with S. My solutions and Martijn's read the input from 
    # differently named files (for example: 'input.txt' vs 'year_2024_day_3.txt')
    normalizedcode= re.sub(r"open\((['\"]).*?\1", "open(S", normalizedcode)
    normalizedcode = re.sub(r'\s+', '', normalizedcode)  # remove all whitespace
    return normalizedcode

def load_external_runtimes(runtime_file):
    # Build a lookup from normalized-code -> runtime (in seconds) using the
    # external timed dataset. Keep only numeric runtimes (drop "None"),
    # and take the median when several timed solutions share one skeleton.
    with open(runtime_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Ggoup all runtimes by their code skeleton
    runtimes_by_key = {}
    for item in data:
        runtime = item['runtime']
        # skip non-numeric runtimes like None
        if not isinstance(runtime, (int, float)):
            continue
        key = normalize_code(item['code'])
        if key not in runtimes_by_key:
            runtimes_by_key[key] = []
        runtimes_by_key[key].append(runtime)
    # Take the median runtime per skeleton
    lookup = {}
    for key, runtimes in runtimes_by_key.items():
        lookup[key] = np.median(runtimes)
    return lookup

def attach_runtimes(clustered_df, runtime_file):
    # Attach the runtime found in the final_data_v1.json dataset to each clustered solution by code skeleton.
    rt = load_external_runtimes(runtime_file)
    df = clustered_df.copy()
    df['runtime'] = df['code'].apply(normalize_code).map(rt)
    matched = df.dropna(subset=['runtime']).copy()
    print(f"Clustered solutions: {len(df)}")
    print(f"Matched with an external runtime: {len(matched)} "
          f"({len(matched) / len(df):.1%})")
    return matched

def efficiency_by_persona(matched_df, persona_names=None):
    # Summarise runtime per persona. Median is the most important number. The mean and max is reported alongside 
    # so the effects of outliers are also visible. 
    # p90 shows the runtime at the 90th percentile
    rows = []
    for cluster, group in matched_df.groupby('cluster'):
        runtimes = group['runtime']
        # use the persona name if one was given, otherwise the cluster number
        if persona_names:
            name = persona_names.get(cluster, cluster)
        else:
            name = cluster
        # round the numbers to 6 decimals as runtimes are in seconds 
        rows.append({
            'persona': name,
            'n': len(runtimes),
            'median': runtimes.median(),
            'mean': runtimes.mean(),
            'p90': runtimes.quantile(0.90),
            'max': runtimes.max()
        })
    summary = pd.DataFrame(rows).set_index('persona')
    print("\nRuntime by persona (seconds):")
    print(summary)
    return summary

def run_efficiency_analysis(clustered_df, runtime_file, persona_names=None):
    # One call to attach runtimes and print the per-persona summary
    matched = attach_runtimes(clustered_df, runtime_file)
    efficiency_by_persona(matched, persona_names)
    return matched