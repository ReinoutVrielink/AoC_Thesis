import json
import pandas as pd
from clustering import run_kmeans

def main():
    with open('data/combined_features.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df_day1 = df[df['day'] == 1]
    first_row = df_day1.iloc[0]
    print(first_row)
    #print("\nComplexity metrics")
    #print(df[['puzzle', 'cyclomatic_complexity', 'avg_function_complexity', 'num_functions', 'sol_id']].to_string(index=False))
    #print("\nHalstead metrics + maintainability index")
    #print(df[['puzzle', 'volume', 'difficulty', 'effort', 'bugs', 'vocabulary', 'maintainability_index', 'sol_id']].to_string(index=False))
    #print("\nRaw metrics")
    #print(df[['puzzle', 'loc', 'lloc', 'sloc', 'comments_count', 'multi_strings', 'blank_lines', 'single_comments', 'sol_id']].to_string(index=False))
    #print("\n AST features")
    #print(df_ast[['puzzle', 'sol_id', 'num_variables', 'avg_variable_name_length']].to_string(index=False))
    clustered_day1 = run_kmeans(df_day1, n_clusters=2)
if __name__ == "__main__":
    main()