import json
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from feature_analysis import extract_all_features, extract_radon_features, extract_ast_features, extract_lexical_features, combine_astandradon_features
import ast
from clustering import run_kmeans, get_feature_columns, cluster_embeddings_per_puzzle
import numpy as np
from GraphCodeBert import load_embeddings, generate_all_embeddings
from clustering import cluster_embeddings_hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 20/4/2026
# Hallo! Dit is mijn eerste file. In deze file inspecteer ik de data om te kijken waarmee ik aan het werken ben


# Ik kijk eerst even naar de solutions.json file, die in de drive van onze scriptie stond
# Ik gebruik hier pandas voor, dit hebben we veel gebruikt in onze studie en is efficiënt
#with open('data/solutions.json', 'r') as f:
#    df = pd.DataFrame(json.load(f))

# Ik wil even kijken hoeveel files er zijn in welke jaren, en hoeveel daarvan in Python geschreven zijn
# Eerst wil ik even checken welke talen er überhaupt in de solutions.json file zitten
#print("Alle talen in de dataset:")
#for taal in sorted(df['language'].unique().astype(str)):
#    print(taal)
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
# Ik hoef dus niet te filteren! (9-5-2026: hier ben ik later op teruggekomen, er zitten ook oplossingen in python 2 syntax, wat problemen geeft bij het parsen van de code)

# 23-4/2026 - Heb net de preprocessing.py file geschreven, waarin ik de solutions.json file heb opgeschoond en opgesplitst in individuele oplossingen (part1/part2)
# lege oplossingen heb ik verwijderd. Ik sla deze op in preprocessed_solutions.json. 
# Nu wil ik even kijken hoeveel oplossingen er in deze nieuwe file zitten, om te checken of het preprocessen goed is gegaan
#with open('data/solutions_with_features.json', 'r') as f:
#    solutions_with_features = json.load(f)
#preprocessed_df = pd.DataFrame(solutions_with_features)
#print(len(df))
#print(len(preprocessed_df['sol_id']))

# there are 7199 solutions in the preprocessed file after splitting part1/part2 and removing empty solutions
# before preprocessing there were 6163 entries

# 29/4/2026 - inspecting the combined radon, ast, and custom features file
#pd.set_option('display.max_columns', None) # first time i printed the metadata for first 10 solution it was cut off
#full_data = extract_all_features()
#full_df = pd.DataFrame(full_data)
#full_df['puzzle'] = 'Day ' + full_df['day'].astype(str)
"""selected_features = [
    'puzzle',
    'sol_id',
    'source_code_lines',
    'max_line_length',
    'total_lines',
    'unique_identifiers',
    'volume',
    'parenthesis_ratio',
    'avg_string_literal_len',
    'for_loop_count',
    'avg_line_length',
    'avg_identifier_length'
]

this is the output of the code above:
extracted stylistic features for 7199 solutions
puzzle                        sol_id  source_code_lines  max_line_length  total_lines  unique_identifiers     volume  parenthesis_ratio  avg_string_literal_len  for_loop_count  avg_line_length  avg_identifier_length
 Day 1          masasin_2015_1_part1                 13               60           17                11.0  68.532389           0.043764                9.571429             1.0        25.941176               7.142857
 Day 1       TheKrumpet_2015_1_part1                  2               35            2                 1.0   4.754888           0.177778                1.000000             0.0        22.000000               1.000000
 Day 2           stuque_2015_2_part1                 21               40           23                 NaN        NaN           0.056940                     NaN             NaN        23.478261                    NaN
 Day 2        [deleted]_2015_2_part1                 96               75          129                27.0  79.954453           0.026991              267.800000             2.0        28.302326               6.803279
 Day 2           jgomo3_2015_2_part1                 14               64           19                22.0  27.000000           0.073218                7.000000             1.0        26.368421               4.771429
 Day 2      streetster__2015_2_part1                 14               69           20                 NaN        NaN           0.045455                     NaN             NaN        25.450000                    NaN
 Day 2 RedditWithBoners_2015_2_part1                  8               89           13                14.0 150.117300           0.076190                1.666667             0.0        23.307692               2.545455
 Day 2        sleepyams_2015_2_part1                 25               72           34                 NaN        NaN           0.054795                     NaN             NaN        18.352941                    NaN
 Day 2        sleepyams_2015_2_part2                 25               72           34                 NaN        NaN           0.054795                     NaN             NaN        18.352941                    NaN
 Day 2          masasin_2015_2_part1                 13               62           18                16.0 220.078200           0.051376               14.500000             1.0        29.333333               4.837838

 We can see that some solutions have NaN values for unique_identifiers and volume, which indicates that there was an issue with parsing those solutions
"""

""""
Example of a solution that failed:
def day2_1():
    total = 0
    for line in open('day2input.txt'):
        l, w, h = line.split('x')
        l, w, h = int(l), int(w), int(h)
        area = 2*l*w + 2*w*h + 2*h*l
        slack = min(l*w, w*h, h*l)
        total += area + slack
    print total

def day2_2():
    total = 0
    for line in open('day2input.txt'):
        l, w, h = line.split('x')
        l, w, h = int(l), int(w), int(h)
        ribbon = 2 * min(l+w, w+h, h+l)
        bow = l*w*h
        total += ribbon + bow
    print total

if __name__ == '__main__':
    day2_1()
    day2_2()

when we look at this code we see that it does 'print total' instead of 'print(total)', which is a syntax error in Python 3
this is likely why the parsing failed and we got NaN values
"""

# This returns the number of rows that have at least one missing value
#total_broken_rows = full_df.isna().any(axis=1).sum()

#print(f"Total solutions with at least one missing value: {total_broken_rows} out of {len(full_df)}")
"""
Total solutions with at least one missing value: 1556 out of 6983
"""
# i'll remove the rows with missing values for now in the preprocessing file, to be able to do some analysis on the remaining data

# I want to try a simple k means clustering on top 10 features to see if there are any interesting clusters in the data, but first I need to normalize the features
# 1-5-2026. testing clustering

#with open('data/combined_features.json', 'r') as f:
#        data = json.load(f)
#df1 = pd.DataFrame(data)
#df1_day = df1[df1['day'] == 3]
#first_row = df1_day.iloc[0]
#
# print(first_row)
#clustered_all = run_kmeans(df1_day, n_clusters=3)
#print("\nCluster distribution:")
#print(clustered_all['cluster'].value_counts().sort_index())

# the clustering wasn't super good but also not very bad. 
# first of all i think i want to focus a bit more on features with more variance.
# also i think the dataset still has some problems because there is quite some data that had NaN values
# I asked Claude to inspect my data a bit more, and as I found out before, there are some solutions that are coded in python 2 syntax. this causes problems
# I also added another code block to inspect the outcomes of the clustering. 
# This is a lot of code (generated by Claude), so I removed it after running. The results of this code are provided in the documentation file of this thesis
# To keep it short the outcome showed that:
# 1. parenthesis_ratio has variance 0.00 (adds noise)
# 2. avg_identifier_length has very low variance (3.00)
# 3. the clustering based on the 10 most effective features in the paper by Sams et al. is not significantly better than clustering on other features 
# 4. there are some other redundant features like source_code_lines (there is already sloc in radon features)
# the following features have very high correlation (above 0.95) and are redundant
# loc <-> total_lines:   1.000
# effort <-> time:   1.000
# volume <-> bugs:   1.000


# Trying clustering on the GraphCodeBERT embeddings
#generate_all_embeddings('data/preprocessed_solutions.json', 'data/graphcodebert_embeddings.json')
# Load embeddings
#print("Loading embeddings.....")
#df = load_embeddings('data/graphcodebert_embeddings.json')
 
#print(f"Loaded {len(df)} total embeddings")
#print(f"Embedding dimension: {len(df['embedding'].iloc[0])}")
"""
# Filter by day (optional)
day_number = 3
df_day = df[df['day'] == day_number]
 
print(f"\n--- Day {day_number} ---")
print(f"Loaded {len(df_day)} embeddings")
print(f"\nFirst few rows:")
print(df_day[['sol_id', 'author', 'day', 'part']].head())
df_day = df[df['day'] == 3]
# Cluster with HDBSCAN
clustered_df = cluster_embeddings_hdbscan(
    df_day, 
    min_cluster_size=8, 
    min_samples=2,
    visualize=True
)
"""


# running clustering on different types of features
from preprocessing import preprocessing
from feature_analysis import extract_all_features
from clusteringanalysis import author_consistency
#preprocessing()
"""
extract_radon_features()
extract_ast_features()
extract_lexical_features()
extract_all_features()
combine_astandradon_features()
with open('data/radon_features.json', 'r') as f:
        radondata = json.load(f)
with open('data/ast_features.json', 'r') as f:
        astdata = json.load(f)
with open('data/lexical_features.json', 'r') as f:
        lexicaldata = json.load(f)
with open('data/combined_features.json', 'r') as f:
        combineddata = json.load(f)
with open('data/astandradon_features.json', 'r') as f:
        astandradon_data = json.load(f)
radondf = pd.DataFrame(radondata)
astdf = pd.DataFrame(astdata)
lexicaldf = pd.DataFrame(lexicaldata)
astandradon_df = pd.DataFrame(astandradon_data)
combineddf = pd.DataFrame(combineddata)
radondf_day3 = radondf[radondf['day'] == 3]
astdf_day3 = astdf[astdf['day'] == 3]
lexicaldf_day3 = lexicaldf[lexicaldf['day'] == 3]
combineddf_day3 = combineddf[combineddf['day'] == 3]
astandradon_df_day3 = astandradon_df[astandradon_df['day'] == 3]
clustered_radon = run_kmeans(radondf, n_clusters=4)
clustered_ast = run_kmeans(astdf, n_clusters=4)
clustered_lexical = run_kmeans(lexicaldf, n_clusters=4)
clustered_combined = run_kmeans(combineddf, n_clusters=4)
clustered_astandradon = run_kmeans(astandradon_df, n_clusters=4)
clustered_astandradon_day3 = run_kmeans(astandradon_df_day3, n_clusters=4)
# also doing some clustering on graphcodebert embeddings
#generate_all_embeddings('data/preprocessed_solutions.json', 'data/graphcodebert_embeddings.json')
# Load embeddings


df = load_embeddings('data/graphcodebert_embeddings.json')
df_day3 = df[df['day'] == 3]
df_day4 = df[df['day'] == 4]
df_day5 = df[df['day'] == 5]
df_day6 = df[df['day'] == 6]
clustered_df = cluster_embeddings_hdbscan(
    df,
    min_cluster_size=20,
    min_samples=3,
    visualize=True
)

df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
# Now run clustering
clustered_df = cluster_embeddings_hdbscan(
    df,
    min_cluster_size=20,
    min_samples=3,
    visualize=True
)

clustered_df_day3 = cluster_embeddings_hdbscan(
    df_day3,
    min_cluster_size=20,
    min_samples=5,
    visualize=True
)

clustered_df_day4 = cluster_embeddings_hdbscan(
    df_day4,
    min_cluster_size=20,
    min_samples=5,
    visualize=True
)
clustered_df_day5 = cluster_embeddings_hdbscan(
    df_day5,
    min_cluster_size=20,
    min_samples=5,
    visualize=True
)
clustered_df_day6 = cluster_embeddings_hdbscan(
    df_day6,
    min_cluster_size=20,
    min_samples=5,
    visualize=True
)
"""

from GraphCodeBert import load_embeddings
from clustering import cluster_embeddings_per_puzzle
from clusteringanalysis import within_puzzle_zscore
from clustering import get_feature_columns
from persona_aggregation import aggregate_local_clusters_to_personas
import pandas as pd, json
import numpy as np

df_emb = load_embeddings('data/graphcodebert_embeddings.json')
df_emb['embedding'] = df_emb['embedding'].apply(lambda x: np.array(x))
df_local = cluster_embeddings_per_puzzle(
    df_emb, min_puzzle_size=15, min_cluster_size=5, min_samples=3,
)
with open('data/combined_features.json') as f:
    df_feat = pd.DataFrame(json.load(f))
features = get_feature_columns(df_feat)
df_norm, _ = within_puzzle_zscore(df_feat, features)

profiles, persona_labels = aggregate_local_clusters_to_personas(
    df_local=df_local,
    df_norm=df_norm,
    features=features,
    k_range=range(2, 9),
    final_k=4,
)