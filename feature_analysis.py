import json
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import ast
import re
# Radon documentation: https://radon.readthedocs.io/en/latest/api.html, https://github.com/rubik/radon

def extract_radon_features():
    with open('data/preprocessed_solutions.json', 'r') as f:
        all_individual_solutions = json.load(f)
    solutions_with_features = []
    for solution in all_individual_solutions:
        code = solution['code']
        solution_with_features = {**solution}
        try:
            # first i used the ComplexityVisitor right from the visitors.py file in Radon, but actually using cc_visit is the right way to do it
            # https://radon.readthedocs.io/en/latest/api.html : this is where the cc_visit is documented, 
            # i used complexityvisitor at first but that is a low level visitor, it's better to use high_level cc_visit
            # Complexity metrics
            complexity_results = cc_visit(code)
            complexities = [block.complexity for block in complexity_results]
            solution_with_features["cyclomatic_complexity"] = sum(complexities)
            solution_with_features["avg_function_complexity"] = sum(complexities) / len(complexities) if complexities else 0
            solution_with_features["num_functions"] = len(complexity_results)
            # Halstead Metrics
            halstead = h_visit(code)
            solution_with_features["h1"] = halstead.total.h1 # number of operators (like +, -, if, for, etc.)
            solution_with_features["h2"] = halstead.total.h2 # number of operands (like variables, constants, etc.)
            solution_with_features["N1"] = halstead.total.N1 # total count of all operators (so including duplicates)
            solution_with_features["N2"] = halstead.total.N2 # total count of all operands (also including duplicates)
            solution_with_features["vocabulary"] = halstead.total.vocabulary # h1 + h2
            solution_with_features["length"] = halstead.total.length # n1 + n2
            solution_with_features["calculated_length"] = halstead.total.calculated_length # an estimate of how long the code should be based on information theory. Dont really know if I will use this later
            solution_with_features["volume"] = halstead.total.volume # estimated amount of information in the code
            solution_with_features["difficulty"] = halstead.total.difficulty # estimated difficulty to write or understand the code
            solution_with_features["effort"] = halstead.total.effort # estimated effort to write or understand the code
            # solution_with_features["time"] = halstead.total.time # Time has a correlation score of 1.00 with effort, so I won't use this one
            # solution_with_features["bugs"] = halstead.total.bugs # 1/5/2026: won't use this one because it has a correlation score of 1.00 with volume
            # maintainability index
            mi_score = mi_visit(code, multi=True)
            solution_with_features["maintainability_index"] = mi_score
            # Raw metrics
            raw_metrics = analyze(code)
            solution_with_features["loc"] = raw_metrics.loc  # Total lines of code
            solution_with_features["lloc"] = raw_metrics.lloc  # Logical lines of code
            solution_with_features["sloc"] = raw_metrics.sloc  # Source lines of code
            solution_with_features["comments_count"] = raw_metrics.comments  # Total comment lines
            solution_with_features["multi_strings"] = raw_metrics.multi  # Multi-line strings (docstrings)
            solution_with_features["blank_lines"] = raw_metrics.blank  # Blank lines
            solution_with_features["single_comments"] = raw_metrics.single_comments  # Lines that are only comments
            # This underneath is feature 5 (defined all features in lexical_analysis.py): Comment to code ratio from the paper by Biel et al. (2023)
            solution_with_features["comment_ratio"] = (raw_metrics.comments + raw_metrics.multi) / raw_metrics.sloc if raw_metrics.sloc > 0 else 0


        except Exception as e:
            # set all metrics to None if any parsing fails
            solution_with_features["cyclomatic_complexity"] = None
            solution_with_features["avg_function_complexity"] = None
            solution_with_features["num_functions"] = None
            solution_with_features["h1"] = None
            solution_with_features["h2"] = None
            solution_with_features["N1"] = None
            solution_with_features["N2"] = None
            solution_with_features["vocabulary"] = None
            solution_with_features["length"] = None
            solution_with_features["calculated_length"] = None
            solution_with_features["volume"] = None
            solution_with_features["difficulty"] = None
            solution_with_features["effort"] = None
            solution_with_features["time"] = None
            solution_with_features["bugs"] = None
            solution_with_features["loc"] = None
            solution_with_features["lloc"] = None
            solution_with_features["sloc"] = None
            solution_with_features["comments_count"] = None
            solution_with_features["multi_strings"] = None
            solution_with_features["blank_lines"] = None
            solution_with_features["single_comments"] = None
            solution_with_features["maintainability_index"] = None
        solutions_with_features.append(solution_with_features)
    
    with open('data/solutions_with_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    
    print(f"Extracted radon features for {len(solutions_with_features)} solutions")
    return solutions_with_features


def extract_ast_features():
    with open('data/preprocessed_solutions.json', 'r') as f:
        all_individual_solutions = json.load(f)
    solutions_with_features = [] 
    for solution in all_individual_solutions:
        code = solution['code']
        solution_with_features = {**solution}
        try:
            tree = ast.parse(code)
            all_nodes = list(ast.walk(tree))
            # variable assignment features
            variable_names = [
                node.id for node in all_nodes 
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            ]
            solution_with_features["num_variables"] = len(variable_names)
            solution_with_features["avg_variable_name_length"] = (
                sum(len(name) for name in variable_names) / len(variable_names) 
                if variable_names else 0
            )
            # identifier features
            identifiers = [node.id for node in all_nodes if isinstance(node, ast.Name)]
            solution_with_features["unique_identifiers"] = len(set(identifiers))
            solution_with_features["avg_identifier_length"] = (
                sum(len(i) for i in identifiers) / len(identifiers) if identifiers else 0
            )
            # string literal features
            string_literals = [
                node.value for node in all_nodes 
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)) or isinstance(node, ast.Str)
            ]
            solution_with_features["avg_string_literal_len"] = (
                sum(len(s) for s in string_literals) / len(string_literals) if string_literals else 0
            )
            # for loop counter
            solution_with_features["for_loop_count"] = len([n for n in all_nodes if isinstance(n, ast.For)])
        except Exception:
            keys = [
                "num_variables", "avg_variable_name_length", "unique_identifiers", 
                "avg_identifier_length", "avg_string_literal_len", "for_loop_count"
            ]
            for key in keys:
                solution_with_features[key] = None
        solutions_with_features.append(solution_with_features)
    with open('data/ast_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    print(f"Extracted AST features for {len(solutions_with_features)} solutions")
    return solutions_with_features

def extract_stylistic_features():
    with open('data/preprocessed_solutions.json', 'r') as f:
        all_individual_solutions = json.load(f)
    # Adding some simpler features that are more about style and formatting
    # These features are inspired by the paper by Biel et al. (2023) and the thesis by Sams et al. (2025)
    # For identifying code stylometry in the thesis by Sams et al. (2025) the top 10 overall important stylometric features consisted of:
    """
    1. The amount of lines that have source code written in them
    2. Amount of characters in longest line in file
    3. Total amount of lines in the file
    4. Amount of unique identifiers (will be done in ast features)
    5. Halstead Volume (already calculated in Radon features)
    6. The number of parenthesis characters per total characters
    7. The average length of string literals in the code (will be done in ast)
    8. Number of loop constructs with the keyword 'for' (will be done in ast)
    9. Average line length in characters
    10. mean size of identifiers (will also be done in ast)
    """
    solutions_with_features = []
    feature_keys = [
        #"source_code_lines",      # 1.
        "max_line_length",        # 2.
        #"total_lines",            # 3.
        #"parenthesis_ratio",      # 6. 
        # 1/5/2026. commented out both parenthesis ratio (variance of 0.00), source code lines and total lines (both redundant)
        "avg_line_length",        # 9.
    ]
    for solution in all_individual_solutions:
        code = solution['code']
        solution_with_features = {**solution}
        try:
            lines = code.splitlines()
            if not lines:
                for key in feature_keys:
                    solution_with_features[key] = 0
            else:
                # Features 2, 3, 9
                line_lengths = [len(l) for l in lines]
                #solution_with_features["total_lines"] = len(lines)
                solution_with_features["max_line_length"] = max(line_lengths)
                solution_with_features["avg_line_length"] = sum(line_lengths) / len(lines)
                # feature 1 - removed this one 1-5-2026 as I realised its redundant (see sloc in radon features)
                #src_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                #solution_with_features["source_code_lines"] = len(src_lines)
                # feature 6
                #if len(code) > 0:
                #    parens_count = code.count('(') + code.count(')')
                #    solution_with_features["parenthesis_ratio"] = parens_count / len(code)
                #else:
                #    solution_with_features["parenthesis_ratio"] = 0
        except Exception as e:
            for key in feature_keys:
                solution_with_features[key] = None
        solutions_with_features.append(solution_with_features)
    with open('data/stylistic_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    print(f"Extracted stylistic features for {len(solutions_with_features)} solutions")
    return solutions_with_features

# Combine all features into one dataset
# I don't think this is the most effective way to do the analysis but for now I keep it
# I did it this way because I just started this script by doing the radon features and ast features in seperate functions
# and then I thought it would be easy to just merge the three datasets together in a separate function
def extract_all_features():
    radon_data = extract_radon_features()
    ast_data = extract_ast_features()
    style_data = extract_stylistic_features()
    combined_map = {}
    def merge_into_map(dataset):
        for entry in dataset:
            sid = entry['sol_id']
            if sid not in combined_map:
                combined_map[sid] = {
                    "sol_id": sid,
                    "author": entry.get("author"),
                    "year": entry.get("year"),
                    "day": entry.get("day"),
                    "part": entry.get("part"),
                    "code": entry.get("code"),
                    "language": entry.get("language")
                }
            combined_map[sid].update(entry)
    merge_into_map(radon_data)
    merge_into_map(ast_data)
    merge_into_map(style_data)
    print("Combined features for all datasets")
    final_dataset = list(combined_map.values())
    with open('data/combined_features.json', 'w') as f:
        json.dump(final_dataset, f)
    return final_dataset

extract_all_features()