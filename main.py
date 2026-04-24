import json
from feature_analysis import extract_radon_features
import pandas as pd

def main():
    extract_radon_features()
    with open('data/solutions_with_features.json', 'r') as f:
        solutions_with_features = json.load(f)
    df = pd.DataFrame(solutions_with_features)
    df = df[df['day'] == 1]
    # Filter out NA values complexities and get first 50
    df = df.dropna(subset=['cyclomatic_complexity', 'avg_function_complexity', 'num_functions']).head(50)
    # Create puzzle column
    df['puzzle'] = 'Day ' + df['day'].astype(str)
    print("\nComplexity metrics")
    print(df[['puzzle', 'cyclomatic_complexity', 'avg_function_complexity', 'num_functions', 'sol_id']].to_string(index=False))
    print("\nHalstead metrics + maintainability index")
    print(df[['puzzle', 'volume', 'difficulty', 'effort', 'bugs', 'vocabulary', 'maintainability_index', 'sol_id']].to_string(index=False))
    print("\nRaw metrics")
    print(df[['puzzle', 'loc', 'lloc', 'sloc', 'comments_count', 'multi_strings', 'blank_lines', 'single_comments', 'sol_id']].to_string(index=False))
if __name__ == "__main__":
    main()