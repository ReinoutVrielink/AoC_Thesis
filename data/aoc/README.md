Advent of Code dataset 2015-2025
================================
The script `collect.py` collects data from all advent of code puzzles.
The answers are produced using solutions from four different repositories,
and verified by submitting them to an advent of code account.

JSON file
---------
The file `puzzles.json` contains a list of all puzzles,
where each puzzle is a dictionary with the following keys:

- `year`: int
- `day`: int
- `title`: str
- `puzzle1`: str; html of the puzzle text of part 1 before anything has been solved.
- `puzzle2`: str; html of the puzzle text of part 1 and 2.
- `puzzle3`; str: html of the puzzle text of part 1, 2 and a message after part 2 has been solved; this is only available in a few cases, such as the last puzzle of the year.
- `input`: str; the input data
- `answer1`: str; the answer for part 1
- `answer2`: str; the answer for part 2 (if any)
- `solver`: str; an ID refererring to the repository that was used to produce the solution
- `solve_time`: float; CPU time in seconds for part 1 and 2 combined.
- `url`: str; adventofcode.com URL for the puzzle
- `unlock_time`: str; release date of puzzle (NB: in `America/New_York` timezone)
- `examples`: list of example extracted from puzzle text; each example is a dict with keys:
  - `input_data`: str
  - `answer1`: str
  - `answer2`: str
  - `extra`: str; extra information(?)
- `easter_eggs`: list of str of the easter eggs in the HTML of the puzzle text.

Separate files
--------------
Some of the data is also offered as separate files.
Given the puzzle of year 2015 day 1, the following files are available:

- `2015/i1`: the input
- `2015/o1`: the expected output; 2 lines of text, with answer 1 and answer 2, respectively. for the last puzzle of the year, there will only be 1 line.
- `2015/p1_0.html`: the puzzle text of part 1 before anything has been solved.
- `2015/p1_1.html`: the puzzle text of part 1 and 2.
- `2015/p1_2.html`: the puzzle text of part 1, 2, and a message after part 2 has been solved; this is only available in a few cases, such as the last puzzle of the year.

