import json

def preprocessing():
    with open('data/solutions.json', 'r') as f:
        data = json.load(f)
    all_individual_solutions = []
    # Transform raw Reddit JSON into cleaned dataset, split part1/part2 into separate entries, 
    # remove empty solutions and Reddit metadata (score, permalink, created_utc), 
    # keep only author/year/day/part/code, and add unique id for solutions
    for entry in data:
        # Split the solutions in part1 and part2
        parts = [("part1", entry.get("part1", "")), ("part2", entry.get("part2", ""))]
        for part_name, code_content in parts:
            if not code_content.strip(): # if the code is empty or just whitespace, skip it
                continue

            solution_metadata = {
                "author": entry["author"],
                "year": entry["year"],
                "day": entry["day"],
                "part": part_name,
                "code": code_content,
                "sol_id": f"{entry['author']}_{entry['year']}_{entry['day']}_{part_name}"
            } 
            all_individual_solutions.append(solution_metadata)
    return all_individual_solutions


all_individual_solutions = preprocessing()
with open('data/preprocessed_solutions.json', 'w') as f:
    json.dump(all_individual_solutions, f)