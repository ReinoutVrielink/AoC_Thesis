import json
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import ast
import re
import tokenize
from io import BytesIO
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
            # 5/5/2026: simplified feature set for clustering by removing redundant Halstead and LOC metrics to reduce multicollinearity and improve interpretability
            # Complexity metrics
            complexity_results = cc_visit(code)
            complexities = [block.complexity for block in complexity_results]

            solution_with_features["cyclomatic_complexity"] = sum(complexities)
            solution_with_features["avg_function_complexity"] = sum(complexities) / len(complexities) if complexities else 0
            solution_with_features["num_functions"] = len(complexity_results)

            # Halstead Metrics
            halstead = h_visit(code)

            solution_with_features["h1"] = halstead.total.h1
            solution_with_features["h2"] = halstead.total.h2
            #solution_with_features["N1"] = halstead.total.N1
            #solution_with_features["N2"] = halstead.total.N2
            #solution_with_features["vocabulary"] = halstead.total.vocabulary
            #solution_with_features["length"] = halstead.total.length
            #solution_with_features["calculated_length"] = halstead.total.calculated_length
            #solution_with_features["volume"] = halstead.total.volume
            solution_with_features["difficulty"] = halstead.total.difficulty
            #solution_with_features["effort"] = halstead.total.effort

            # maintainability index
            mi_score = mi_visit(code, multi=False)
            solution_with_features["maintainability_index"] = mi_score

            # Raw metrics
            raw_metrics = analyze(code)

            solution_with_features["sloc"] = raw_metrics.sloc  # Source lines of code

            #solution_with_features["loc"] = raw_metrics.loc
            #solution_with_features["lloc"] = raw_metrics.lloc

            solution_with_features["comments_count"] = raw_metrics.comments
            solution_with_features["multi_strings"] = raw_metrics.multi
            solution_with_features["blank_lines"] = raw_metrics.blank
            solution_with_features["single_comments"] = raw_metrics.single_comments

            # removed both avg line length and max line length. they are both already in the lexical features

            # This underneath is feature 5 (defined all features in lexical_analysis.py): Comment to code ratio from the paper by Biel et al. (2023)
            solution_with_features["comment_ratio"] = (
                raw_metrics.comments + raw_metrics.multi
            ) / raw_metrics.sloc if raw_metrics.sloc > 0 else 0

        except Exception as e:
            # set all metrics to None if any parsing fails
            solution_with_features["cyclomatic_complexity"] = None
            solution_with_features["avg_function_complexity"] = None
            solution_with_features["num_functions"] = None

            solution_with_features["h1"] = None
            solution_with_features["h2"] = None
            #solution_with_features["N1"] = None
            #solution_with_features["N2"] = None
            #solution_with_features["vocabulary"] = None
            #solution_with_features["length"] = None
            #solution_with_features["calculated_length"] = None
            #solution_with_features["volume"] = None
            solution_with_features["difficulty"] = None
            #solution_with_features["effort"] = None

            solution_with_features["sloc"] = None
            #solution_with_features["loc"] = None
            #solution_with_features["lloc"] = None

            solution_with_features["comments_count"] = None
            solution_with_features["multi_strings"] = None
            solution_with_features["blank_lines"] = None
            solution_with_features["single_comments"] = None

            solution_with_features["maintainability_index"] = None

        solutions_with_features.append(solution_with_features)

    with open('data/radon_features.json', 'w') as f:
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

            # Added function that calculates nesting depth
            def get_max_nesting_depth(node, depth=0):
                if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                    depth += 1
                max_depth = depth
                for child in ast.iter_child_nodes(node):
                    child_depth = get_max_nesting_depth(child, depth)
                    max_depth = max(max_depth, child_depth)
                return max_depth

            solution_with_features["max_nesting_depth"] = get_max_nesting_depth(tree)

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
                "avg_identifier_length", "avg_string_literal_len", "for_loop_count",
                "max_nesting_depth"
            ]
            for key in keys:
                solution_with_features[key] = None
        solutions_with_features.append(solution_with_features)
    with open('data/ast_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    print(f"Extracted AST features for {len(solutions_with_features)} solutions")
    return solutions_with_features

def extract_lexical_features():
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
        "total_lines",            # 3.
        #"parenthesis_ratio",      # 6. 
        # 1/5/2026. commented out both parenthesis ratio (variance of 0.00), source code lines and total lines (both redundant)
        #"avg_line_length",        # 9. drop bcs of high correlation
    ]
    # 4/5/2026: I also added a bunch of token-level stylometry features that are inspired by the paper by Biel et al. (2023) and the thesis by Sams et al. (2025)
    # these features focus on keyword preferences, operator patterns, and built-in function usage
    keywords = {'if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'try', 
                 'class', 'lambda', 'return', 'yield', 'import', 'from', 'as', 'with', 'and', 'or', 'not'}
    builtins = {'range', 'enumerate', 'zip', 'map', 'filter', 'len', 'sum', 'min', 'max', 'sorted', 
                'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'print', 'input', 'all', 'any'}
    arithmetic_ops = {'+', '-', '*', '/', '//', '%', '**'}
    comparison_ops = {'==', '!=', '<', '>', '<=', '>='}

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
                solution_with_features["total_lines"] = len(lines)
                solution_with_features["max_line_length"] = max(line_lengths)
                #solution_with_features["avg_line_length"] = sum(line_lengths) / len(lines)
            # Token-level stylometry extraction
            lines_of_code = len(lines)
            kw_counts = {kw: 0 for kw in keywords}
            builtin_counts = {bf: 0 for bf in builtins}
            arith_count, comp_count = 0, 0
            total_kw, total_ops, total_bf = 0, 0, 0
            try:
                tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
                for tok in tokens:
                    token_string = tok.string
                    # keyword detection
                    if token_string in keywords:
                        kw_counts[token_string] += 1
                        total_kw += 1
                    # builtin usage
                    elif token_string in builtins:
                        builtin_counts[token_string] += 1
                        total_bf += 1
                    # operators
                    elif token_string in arithmetic_ops:
                        arith_count += 1
                        total_ops += 1
                    elif token_string in comparison_ops:
                        comp_count += 1
                        total_ops += 1
            except tokenize.TokenError:
                pass
            # storing keyword counts and ratios
            for kw, count in kw_counts.items():
                solution_with_features[f'kw_{kw}'] = count
            if total_kw > 0:
                conditionals = kw_counts['if'] + kw_counts['elif'] + kw_counts['else']
                loops = kw_counts['for'] + kw_counts['while']
                #functions = kw_counts['def'] + kw_counts['lambda']
                solution_with_features['conditional_preference'] = conditionals / (conditionals + loops) if (conditionals + loops) > 0 else 0
                solution_with_features['for_over_while'] = kw_counts['for'] / loops if loops > 0 else 0
                #solution_with_features['lambda_ratio'] = kw_counts['lambda'] / functions if functions > 0 else 0
                #solution_with_features['error_handling_ratio'] = (kw_counts['try'] + kw_counts['except']) / total_kw
            else:
                solution_with_features['conditional_preference'] = 0
                solution_with_features['for_over_while'] = 0
                #solution_with_features['lambda_ratio'] = 0
                #solution_with_features['error_handling_ratio'] = 0
            # storing built-in counts and ratios
            for bf, count in builtin_counts.items():
                solution_with_features[f'builtin_{bf}'] = count
            if total_bf > 0:
                iteration_bf = builtin_counts['range'] + builtin_counts['enumerate'] + builtin_counts['zip'] + builtin_counts['map'] + builtin_counts['filter']
                solution_with_features['iteration_builtins_ratio'] = iteration_bf / total_bf
                solution_with_features['enumerate_over_range'] = builtin_counts['enumerate'] / (builtin_counts['enumerate'] + builtin_counts['range']) if (builtin_counts['enumerate'] + builtin_counts['range']) > 0 else 0
            else:
                solution_with_features['iteration_builtins_ratio'] = 0
                solution_with_features['enumerate_over_range'] = 0
            # Operator ratios
            if total_ops > 0:
                solution_with_features['op_arithmetic_ratio'] = arith_count / total_ops
                solution_with_features['op_comparison_ratio'] = comp_count / total_ops
            else:
                solution_with_features['op_arithmetic_ratio'] = 0
                solution_with_features['op_comparison_ratio'] = 0
            # Token density
            solution_with_features['keywords_per_line'] = total_kw / lines_of_code if lines_of_code > 0 else 0
            solution_with_features['builtins_per_line'] = total_bf / lines_of_code if lines_of_code > 0 else 0
        except Exception as e:
            for key in feature_keys:
                solution_with_features[key] = None
            for kw in keywords:
                solution_with_features[f'kw_{kw}'] = None
            for bf in builtins:
                solution_with_features[f'builtin_{bf}'] = None
            for key in ['conditional_preference', 'for_over_while',
                       'iteration_builtins_ratio', 'enumerate_over_range', 'op_arithmetic_ratio', 
                       'op_comparison_ratio', 'keywords_per_line', 'builtins_per_line']:
                solution_with_features[key] = None
        solutions_with_features.append(solution_with_features)
    with open('data/lexical_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    print(f"Extracted lexical features for {len(solutions_with_features)} solutions")
    return solutions_with_features

# Combine all features into one dataset
# I don't think this is the most effective way to do the analysis but for now I keep it
# I did it this way because I just started this script by doing the radon features and ast features in seperate functions
# and then I thought it would be easy to just merge the three datasets together in a separate function
def extract_all_features():
    radon_data = extract_radon_features()
    ast_data = extract_ast_features()
    style_data = extract_lexical_features()
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

def combine_astandradon_features():
    radon_data = extract_radon_features()
    ast_data = extract_ast_features()
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
    print("Combined radon and AST features")
    astandradon_dataset = list(combined_map.values())
    with open('data/astandradon_features.json', 'w') as f:
        json.dump(astandradon_dataset, f)
    return astandradon_dataset