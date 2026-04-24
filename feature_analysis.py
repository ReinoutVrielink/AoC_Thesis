import json
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import ast
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
            solution_with_features["time"] = halstead.total.time # estimated time to program in seconds. Assumption is 18 seconds per mental operation
            solution_with_features["bugs"] = halstead.total.bugs # estimated number of bugs in the code
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
            # parsing code into ast tree. just want to look if it works so i only extract two simple features: number of variables and average variable name length. Will expand this later
            tree = ast.parse(code)
            variable_names = [
                node.id for node in ast.walk(tree) 
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            ]
            solution_with_features["num_variables"] = len(variable_names)
            solution_with_features["avg_variable_name_length"] = (
                sum(len(name) for name in variable_names) / len(variable_names) 
                if variable_names else 0
            )
        except Exception as e:
            solution_with_features["num_variables"] = None
            solution_with_features["avg_variable_name_length"] = None
            
        solutions_with_features.append(solution_with_features)
    
    # Save to a different file first, might concatenate them later
    with open('data/ast_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
        
    print(f"Extracted AST features for {len(solutions_with_features)} solutions")
    return solutions_with_features
            