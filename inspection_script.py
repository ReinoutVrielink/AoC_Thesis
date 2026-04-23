import json
import pandas as pd
# 20/4/2026
# Hallo! Dit is mijn eerste file. In deze file inspecteer ik de data om te kijken waarmee ik aan het werken ben


# Ik kijk eerst even naar de solutions.json file, die in de drive van onze scriptie stond
# Ik gebruik hier pandas voor, dit hebben we veel gebruikt in onze studie en is efficiënt
with open('data/solutions.json', 'r') as f:
    df = pd.DataFrame(json.load(f))

# Ik wil even kijken hoeveel files er zijn in welke jaren, en hoeveel daarvan in Python geschreven zijn
# Eerst wil ik even checken welke talen er überhaupt in de solutions.json file zitten
print("Alle talen in de dataset:")
for taal in sorted(df['language'].unique().astype(str)):
    print(taal)
# Dit is de output:
"""
Alle talen in de dataset:
**Python 3.12**
PYTHON
PYTHON + PANDAS
PYTHON3
Python
Python
I solved it in a longer winded way first, then stuffed it all in a list comprehension for amusement in a way that I'm not proud of. But hey, it works.
    list_comp_pt_1 = sum(
        [
            int(x[0
Python & pandas
Python + "paper & pen"
Python + NumPy
Python + SciPy
Python + Scribbling on looseleaf
Python + Z3
Python - VERY BEGGINER
Python / Math
Python 3
Python 3, Typescript, Go
Python 3.11
Python 3.11.5
Python 3.12
Python 3.8
Python Golf
Python and C++
Python golf
Python with NumPy
Python with SymPy
Python+Brain
Python, NumPy
Python, Part 2
Python, SageMath
Python/Codon
Python/Julia
Python3
Python3, shapely
Python3.8
Python3.8+
Python:
Standard Python 3.9
[Python
python
python 3
python3
python3.12
z3 (& Python)
"""

# Als ik kijk naar deze output zie ik dat de oplossingen eigenlijk allemaal in python zijn
# Ik hoef dus niet te filteren!

# 23-4/2026 - Heb net de preprocessing.py file geschreven, waarin ik de solutions.json file heb opgeschoond en opgesplitst in individuele oplossingen (part1/part2)
# lege oplossingen heb ik verwijderd. Ik sla deze op in preprocessed_solutions.json. 
# Nu wil ik even kijken hoeveel oplossingen er in deze nieuwe file zitten, om te checken of het preprocessen goed is gegaan
with open('data/solutions_with_features.json', 'r') as f:
    solutions_with_features = json.load(f)
preprocessed_df = pd.DataFrame(solutions_with_features)
print(len(df))
print(len(preprocessed_df['sol_id']))

# there are 7199 solutions in the preprocessed file after splitting part1/part2 and removing empty solutions
# before preprocessing there were 6163 entries