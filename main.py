import json
from feature_analysis import extract_all_features
import pandas as pd

def main():
    extract_all_features()
    """
    #extract_ast_features()
    with open('data/solutions_with_features.json', 'r') as f:
        solutions_with_features = json.load(f)
    with open('data/ast_features.json', 'r') as f:
        ast_data = json.load(f)
    df = pd.DataFrame(solutions_with_features)
    df_ast = pd.DataFrame(ast_data)
    df = df.dropna()
    df_ast = df_ast.dropna()
    df = df[df['day'] == 1]
    df_ast = df_ast[df_ast['day'] == 1]
    df_ast['puzzle'] = 'Day ' + df_ast['day'].astype(str)
    # Filter out NA values complexities and get first 50
    df = df.head(50)
    # Create puzzle column
    df['puzzle'] = 'Day ' + df['day'].astype(str)
    """
    #print("\nComplexity metrics")
    #print(df[['puzzle', 'cyclomatic_complexity', 'avg_function_complexity', 'num_functions', 'sol_id']].to_string(index=False))
    #print("\nHalstead metrics + maintainability index")
    #print(df[['puzzle', 'volume', 'difficulty', 'effort', 'bugs', 'vocabulary', 'maintainability_index', 'sol_id']].to_string(index=False))
    #print("\nRaw metrics")
    #print(df[['puzzle', 'loc', 'lloc', 'sloc', 'comments_count', 'multi_strings', 'blank_lines', 'single_comments', 'sol_id']].to_string(index=False))
    #print("\n AST features")
    #print(df_ast[['puzzle', 'sol_id', 'num_variables', 'avg_variable_name_length']].to_string(index=False))
if __name__ == "__main__":
    main()