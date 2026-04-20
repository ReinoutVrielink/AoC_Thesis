"""Submit puzzle answer to Advent of Code and collect puzzle data.

This script expects an AOC_SESSION environment variable,
see https://github.com/wimglenn/advent-of-code/issues/1"""
import io
import os
import sys
import json
import time
from contextlib import redirect_stdout
import aocd
from tqdm import tqdm


def get(year, day, skipsolved=True):
	"""Collect all data for a single puzzle.

	:param skipsolved: whether to skip solved problems; pass False to re-run
			all solutions in order to collect new benchmarks."""
	puzzle = aocd.get_puzzle(year=year, day=day)
	answer1 = answer2 = solver = solve_time = None
	lastday = 12 if year >= 2025 else 25
	if skipsolved and puzzle.answered_a and (
			puzzle.answered_b or day == lastday):
		answer1 = puzzle.answer_a
		answer2 = puzzle.answer_b
		print(f'{year=} {day=} ALREADY SOLVED {answer1=} {answer2=}',
				file=sys.stderr)
	else:
		if answer1 is None:
			for solver, solve_func in SOLVERS.items():
				if (solver, year, day) in TOOSLOW:
					continue
				start = time.perf_counter()
				try:
					answer1, answer2 = solve_func(year, day, puzzle.input_data)
				except Exception as err:
					print(f'ERROR {year=} {day=} {solver}: {err}',
							file=sys.stderr)
					continue
				finally:
					os.chdir(CWD)
				solve_time = time.perf_counter() - start
				print(f'{year=} {day=} {solver}: {answer1=} {answer2=} '
						f'{solve_time=:.2f}', file=sys.stderr)
				if answer1 is None or (answer2 is None and day != lastday):
					continue
				if (aocd.utils.coerce(answer1) != puzzle.answer_a
						or (day != lastday
							and aocd.utils.coerce(answer2) != puzzle.answer_b)):
					print(f'WRONG {year=} {day=} {solver}', file=sys.stderr)
					continue
				if answer1 is not None and answer1 != '':
					response = aocd.submit(answer1, year=year, day=day,
							part='a', reopen=False)
					if response is not None:
						time.sleep(5)
					# last day has no part 2; only try part 2 if part 1 is solved
					if (answer2 is not None and answer2 != ''
							and puzzle.answered_a
							and answer1 != answer2):
						response = aocd.submit(answer2, year=year, day=day,
								part='b', reopen=False)
						if response is not None:
							time.sleep(5)
					if puzzle.answered_a and (day == lastday or puzzle.answered_b):
						break
					print('waiting 1 minute to try again', file=sys.stderr)
					time.sleep(60)
		if answer1 is None:
			print(f'{year=} {day=} NO ANSWER', file=sys.stderr)
	with open(puzzle.prose0_path, encoding='utf8') as inp:
		puzzle1 = inp.read()
	try:
		with open(puzzle.prose1_path, encoding='utf8') as inp:
			puzzle2 = inp.read()
	except FileNotFoundError:
		puzzle2 = None
	try:
		with open(puzzle.prose2_path, encoding='utf8') as inp:
			puzzle3 = inp.read()
	except FileNotFoundError:
		puzzle3 = None
	if not os.path.exists(str(year)):
		os.mkdir(str(year))
	with open(f'{year}/i{day}', 'w', encoding='utf8') as out:
		out.write(puzzle.input_data)
	for n, puztext in enumerate([puzzle1, puzzle2, puzzle3]):
		if puztext is not None:
			with open(f'{year}/p{day}_{n}.html', 'w', encoding='utf8') as out:
				out.write(puztext)
	if puzzle.answered_a and puzzle.answered_b:
		with open(f'{year}/o{day}', 'w', encoding='utf8') as out:
			print(puzzle.answer_a, file=out)
			print(puzzle.answer_b, file=out)
	elif day == lastday and puzzle.answered_a:
		with open(f'{year}/o{day}', 'w', encoding='utf8') as out:
			print(puzzle.answer_a, file=out)
	return {'year': year,
			'day': day,
			'title': puzzle.title,
			'puzzle1': puzzle1,
			'puzzle2': puzzle2,
			'puzzle3': puzzle3,
			'input': puzzle.input_data,
			'answer1': getattr(puzzle, 'answer_a', None),
			'answer2': None if day == lastday
				else getattr(puzzle, 'answer_b', None),
			'solver': solver,
			'solve_time': solve_time,
			'url': puzzle.url,
			'unlock_time': str(puzzle.unlock_time(local=False)),
			'examples': [{
					'input_data': ex.input_data,
					'answer1': ex.answer_a,
					'answer2': ex.answer_b,
					'extra': ex.extra}
					for ex in puzzle.examples],
			'easter_eggs': [str(a) for a in puzzle.easter_eggs],
			}


def solve_fuglede(year, day, input_data):
	"""Solve puzzle using https://github.com/fuglede/adventofcode"""
	path = f'/home/andreas/src/fuglede-AoC/{year}/day{day:02}'
	os.chdir(path)
	input_path = 'input'
	with open(input_path, 'w', encoding='utf8') as out:
		out.write(input_data)
	try:
		os.mkdir(f'day{day:02}')
	except FileExistsError:
		pass
	input_path2 = f'day{day:02}/input'
	with open(input_path2, 'w', encoding='utf8') as out:
		out.write(input_data)
	# Python standard library also has a "code" module,
	# so our path must go first.
	# clear import cache because we import a different module with the same name
	if 'solutions' in sys.modules:
		del sys.modules['solutions']
	try:
		sys.path.insert(0, path)
		sys.path.insert(0, f'/home/andreas/src/fuglede-AoC/{year}')
		# the answers are print()ed to stdout,
		# we assume the first two lines of output are the answers
		output = io.StringIO()
		with redirect_stdout(output):  # https://stackoverflow.com/a/40984270
			# unused import but part1() and part2() are called on import
			import solutions
			assert solutions.__name__ == 'solutions'  # silnce unused important warning
	finally:
		sys.path.pop(0)
		sys.path.pop(0)
		if 'vm' in sys.modules:
			del sys.modules['vm']
	result1 = result2 = None
	lines = output.getvalue().splitlines()
	if len(lines):
		result1 = lines[0].strip()
	if len(lines) > 1:
		result2 = lines[1].strip()
	os.unlink(input_path)
	os.chdir(CWD)
	return result1, result2


def solve_AbdeI1(year, day, input_data):
	"""Solve puzzle using https://github.com/AbdeI1/AdventOfCode"""
	path = f'/home/andreas/src/AbdeI1-AoC/{year}/{day:02}'
	input_path = f'{path}/input.txt'
	with open(input_path, 'w', encoding='utf8') as out:
		out.write(input_data)
	# Python standard library also has a "code" module,
	# so our path must go first.
	# clear import cache because we import a different module with the same name
	if 'code' in sys.modules:
		del sys.modules['code']
	try:
		sys.path.insert(0, path)
		sys.path.insert(0, f'/home/andreas/src/AbdeI1-AoC/{year}')
		os.chdir(path)
		# the answers are print()ed to stdout,
		# we assume the first two lines of output are the answers
		output = io.StringIO()
		with redirect_stdout(output):  # https://stackoverflow.com/a/40984270
			# unused import but part1() and part2() are called on import
			import code
			assert code.__name__ == 'code'  # silnce unused important warning
	finally:
		sys.path.pop(0)
		sys.path.pop(0)
		os.chdir(CWD)
		if 'intcode' in sys.modules:
			del sys.modules['intcode']
		if '..' in sys.path:
			sys.path.remove('..')
	result1 = result2 = None
	lines = output.getvalue().splitlines()
	if len(lines):
		result1 = lines[0].strip()
	if len(lines) > 1:
		result2 = lines[1].strip()
	os.unlink(input_path)
	return result1, result2


def solve_fadi(year, day, input_data):
	"""Solve puzzle using https://github.com/Fadi88/AoC"""
	path = f'/home/andreas/src/Fadi88-AoC/{year}/day{day:02}'
	# Python standard library also has a "code" module,
	# so our path must go first.
	# clear import cache because we import a different module with the same name
	if 'code' in sys.modules:
		del sys.modules['code']
	try:
		sys.path.insert(0, path)
		import code
	finally:
		sys.path.pop(0)
		if 'intcode' in sys.modules:
			del sys.modules['intcode']

	input_path = 'input.txt'  # will be written to current working directory
	with open(input_path, 'w', encoding='utf8') as out:
		out.write(input_data)
	input_path2 = f'day{day:02}'  # input path is inconsistent
	try:
		os.mkdir(input_path2)
	except FileExistsError:
		pass
	with open(input_path2 + '/' + input_path, 'w', encoding='utf8') as out:
		out.write(input_data)

	# the answers are print()ed to stdout,
	# we assume the first line of output is the answer
	output = io.StringIO()
	with redirect_stdout(output):  # https://stackoverflow.com/a/40984270
		if hasattr(code, 'part1'):
			code.part1()
		elif hasattr(code, 'part_1'):
			code.part_1()
	result1 = None
	out = output.getvalue()
	if out:
		result1 = out.splitlines()[0].strip()

	output = io.StringIO()
	with redirect_stdout(output):
		if hasattr(code, 'part2'):
			code.part2()
		elif hasattr(code, 'part_2'):
			code.part_2()
		out = output.getvalue()
		result2 = out.splitlines()[0].strip() if out else None

	os.unlink(input_path)
	os.unlink(f'{input_path2}/{input_path}')
	os.rmdir(input_path2)
	return result1, result2


def solve_avc(year, day, input_data):
	"""Solve puzzle using https://github.com/andreasvc/adventofcode"""
	path = f'/home/andreas/code/adventofcode/{year}'
	if 'adventofcode' in sys.modules:
		del sys.modules['adventofcode']
	try:
		sys.path.insert(0, path)
		import adventofcode
		if hasattr(adventofcode, f'day{day}'):
			func = getattr(adventofcode, f'day{day}')
			result = func(input_data.rstrip())
			if isinstance(result, tuple) and len(result) == 2:
				result1, result2 = result
			else:
				result1 = result
				result2 = None
		else:
			result1 = result2 = None
			if hasattr(adventofcode, f'day{day}a'):
				func1 = getattr(adventofcode, f'day{day}a')
				result1 = func1(input_data.rstrip())
			if hasattr(adventofcode, f'day{day}b'):
				func1 = getattr(adventofcode, f'day{day}b')
				result2 = func1(input_data.rstrip())
	finally:
		sys.path.pop(0)
		if 'intcode' in sys.modules:
			del sys.modules['intcode']
	return result1, result2


def main():
	"""Collect all years/days."""
	sys.path.insert(0, '/home/andreas/code/adventofcode')  # for common module
	sys.path.insert(0, '/home/andreas/code/adventofcode/2019')  # for intcode module
	puzzles = [(year, day + 1)
			for year in range(2015, 2026)
				for day in range(12 if year >= 2025 else 25)]
	results = []
	for year, day in tqdm(puzzles):
		try:
			result = get(year, day, skipsolved=False)
		except aocd.exceptions.PuzzleLockedError:
			continue
		results.append(result)
	with open('puzzles.json', 'w', encoding='utf8') as out:
		json.dump(results, out, indent=2)


CWD = os.getcwd()
SOLVERS = {
		'avc': solve_avc,
		'fuglede': solve_fuglede,
		'AbdeI1': solve_AbdeI1,
		'fadi': solve_fadi,
		}
TOOSLOW = {
		('fuglede', 2015, 19),
		('avc', 2016, 19),
		('avc', 2016, 22),
		('AbdeI1', 2019, 18),
		('fuglede', 2016, 11),
		# ('fuglede', 2019, 18),
		('fuglede', 2022, 16),
		('fuglede', 2022, 19),
		('fuglede', 2024, 14),
		}

if __name__ == '__main__':
	main()
