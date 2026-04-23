import json
from radon.visitors import ComplexityVisitor
# Radon documentation: https://radon.readthedocs.io/en/latest/api.html

def extract_radon_features():
    with open('data/preprocessed_solutions.json', 'r') as f:
        all_individual_solutions = json.load(f)
    solutions_with_features = []
    for solution in all_individual_solutions:
        code = solution['code']
        solution_with_features = {**solution}
        try:
            v = ComplexityVisitor.from_code(code)
            complexities = [func.complexity for func in v.functions]
            solution_with_features["cyclomatic_complexity"] = sum(complexities)
            solution_with_features["avg_function_complexity"] = sum(complexities) / len(complexities) if complexities else 0
            solution_with_features["num_functions"] = len(v.functions)
            
        except Exception as e:
            solution_with_features["cyclomatic_complexity"] = None
            solution_with_features["avg_function_complexity"] = None
            solution_with_features["num_functions"] = 0
        
        solutions_with_features.append(solution_with_features)
    
    # save to file (will overwrite later with more metrics)
    with open('data/solutions_with_features.json', 'w') as f:
        json.dump(solutions_with_features, f)
    
    print(f"Extracted cyclomatic complexity for {len(solutions_with_features)} solutions")
    return solutions_with_features