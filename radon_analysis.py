import json
from radon.complexity import cc_visit
from radon.metrics import h_visit
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
        except Exception as e:
            # set all metrics to None if any parsing fails
            solution_with_features["cyclomatic_complexity"] = None
            solution_with_features["avg_function_complexity"] = None
            solution_with_features["num_functions"] = 0
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
        solutions_with_features.append(solution_with_features)
    
    with open('data/solutions_with_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    
    print(f"Extracted radon features for {len(solutions_with_features)} solutions")
    return solutions_with_features