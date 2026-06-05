import json
import ast
from collections import defaultdict, Counter

# 5/5/2026: Changed preprocessing function as it let through python2 code and many other invalid solutions
# These invalid solutions resulted in errors and NaN values in the feature extraction phase
def is_valid_python(code):
    # checking already if code is actually valid python code
    # source is https://stackoverflow.com/questions/11854745/how-to-tell-if-a-string-contains-valid-python-code
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def preprocessing():
    # Transform raw Reddit JSON into cleaned dataset, split part1/part2 into separate entries, 
    # remove empty solutions and Reddit metadata (score, permalink, created_utc), 
    # keep only author/year/day/part/code, and add unique id for solutions
    print("preprocessing...")
    with open('data/solutions.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} raw entries from solutions.json")
    all_individual_solutions = []
    skipped_no_metadata = 0
    skipped_empty_code = 0
    skipped_invalid_syntax = 0
    sol_id_counter = defaultdict(int)
    for entry in data:
        required_fields = ['author', 'year', 'day']
        if not all(key in entry for key in required_fields):
            skipped_no_metadata += 1
            continue
        # Process part1 and part2
        parts = [("part1", entry.get("part1", "")), ("part2", entry.get("part2", ""))]
        for part_name, code_content in parts:
            # Skip if code is empty or only whitespace
            if not code_content.strip():
                skipped_empty_code += 1
                continue
            # Skip if code has invalid Python syntax (like python2 syntax that was present in some solutions)
            if not is_valid_python(code_content):
                skipped_invalid_syntax += 1
                continue
            # Create a solution ID with a counter for potential duplicates
            base_id = f"{entry['author']}_{entry['year']}_{entry['day']}_{part_name}"
            sol_id_counter[base_id] += 1
            if sol_id_counter[base_id] > 1:
                sol_id = f"{base_id}_v{sol_id_counter[base_id]}"
            else:
                sol_id = base_id
            # Create solution metadata
            solution_metadata = {
                "author": entry["author"],
                "year": entry["year"],
                "day": entry["day"],
                "language": entry.get("language", "Unknown"),
                "part": part_name,
                "code": code_content,
                "sol_id": sol_id
            }  
            all_individual_solutions.append(solution_metadata)
    # Save preprocessed data to file
    with open('data/preprocessed_solutions.json', 'w') as f:
        json.dump(all_individual_solutions, f)
    print(f"\nSaved preprocessed solutions to 'data/preprocessed_solutions.json'")
    return all_individual_solutions


def validate_preprocessed_data(filepath='data/preprocessed_solutions.json'):
    # Addition at 5/5/2026: validating the preprocessed data for quality and completeness before moving on to feature extraction and clustering
    # *I created this function with the help of Claude
    # not a pipeline step, just run manually once to sanity-check the preprocessed data in the beginning of the project
    print(f"\nValidating preprocessed data from {filepath}...")
    with open(filepath, 'r') as f:
        preprocessed = json.load(f)
    print(f"Total solutions: {len(preprocessed)}")
    # Check for missing required fields
    required_fields = ['author', 'year', 'day', 'code', 'sol_id', 'part']
    missing_counts = {}
    for field in required_fields:
        count = sum(1 for s in preprocessed if not s.get(field))
        missing_counts[field] = count
        if count > 0:
            print(f"Missing '{field}': {count} solutions")
        else:
            print(f"'{field}': present in all solutions")
    # Check code validity
    invalid_code_count = 0
    for sol in preprocessed:
        if not is_valid_python(sol['code']):
            invalid_code_count += 1
    if invalid_code_count > 0:
        print(f"Invalid Python code: {invalid_code_count} solutions")
    else:
        print(f"All code snippets are valid Python")
    # Distribution by day
    days = [s['day'] for s in preprocessed]
    print(f"\n  Solutions per day:")
    day_counts = Counter(days)
    for day in sorted(day_counts.keys())[:10]:  # Show first 10 days
        print(f"    Day {day}: {day_counts[day]} solutions")
    # Distribution by part
    parts = [s['part'] for s in preprocessed]
    part_counts = Counter(parts)
    for part in sorted(part_counts.keys()):
        print(f"    {part}: {part_counts[part]} solutions")
    print("\nValidation complete!")
    return preprocessed

