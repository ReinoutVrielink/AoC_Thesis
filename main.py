import json
from radon_analysis import extract_radon_features
import pandas as pd

def main():
    # Extract radon features
    extract_radon_features()
    # Load solutions with features
    with open('data/solutions_with_features.json', 'r') as f:
        solutions_with_features = json.load(f)
    # Convert to DataFrame
    df = pd.DataFrame(solutions_with_features)
    # Filter out None complexities and get first 50
    df = df[df['cyclomatic_complexity'].notna()].head(50)
    # Create puzzle column
    df['puzzle'] = 'Day ' + df['day'].astype(str)
    print("\n")
    print(df[['author', 'puzzle', 'cyclomatic_complexity', 'sol_id']].to_string(index=False))
    code = df[df['sol_id'] == 'minno_2015_7_part1']['code'].values[0]
    print(code)
if __name__ == "__main__":
    main()