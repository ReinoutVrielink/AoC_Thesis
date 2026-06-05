import json
import multiprocessing
import time

# this entire file was not my work; it was the work of Martijn Huls
# The file is here to demonstrate how the runtimes were measured, but I did not run this code myself
# The runtimes used in the analysis come from an external dataset of timed Advent of Code solutions (final_data_v1.json)
path = "C:\\Users\\hulsm\\Documents\\Scriptie\\data\\runnable_data\\"

def open_data(file_path):
    print(f"Opening data from {file_path}...")
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:5]

def get_puzzle_data(puzzle):
    with open(f"year_{puzzle['year']}_day_{puzzle['day']}.txt", "r", encoding="utf-8") as f:
        return f.read()

def main():
    data = open_data("parsed_solutions.json")
    runtimes = []

    for i in data:
        puzzle_data = get_puzzle_data(i)
        print(puzzle_data)

        code = i["code"]

        process = multiprocessing.Process(target=exec, args=(code,))

        start = time.perf_counter()

        process.start()
        process.join(timeout=180)  # wait for 3 minutes

        if process.is_alive():
            print(
                f"Code for id {i['id']} is taking too long! Terminating..."
            )
            process.terminate()
            process.join()
            execution_time = "Timeout"

        elif process.exitcode != 0:
            print(
                f"Code for id {i['id']} crashed!"
            )
            execution_time = None

        else:
            end = time.perf_counter()
            execution_time = end - start

            print(
                f"Execution time for id {i['id']}: {execution_time} seconds"
            )

        runtimes.append((i['id'], execution_time))

    print("All runtimes:")
    for puzzle_id, runtime in runtimes:
        print(f"Puzzle ID {puzzle_id}: {runtime} seconds")

    completed_puzzles = sum(1 for _, runtime in runtimes if runtime is not None)
    total_puzzles = len(runtimes)
    print(f"Completed puzzles: {completed_puzzles}/{total_puzzles}")
    with open("runtimes.json", "w", encoding="utf-8") as f:
        json.dump(runtimes, f, indent=2)


if __name__ == "__main__":
    main()