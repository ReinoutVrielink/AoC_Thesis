import json
import pandas as pd
from clustering import run_kmeans
from feature_analysis import extract_radon_features
from feature_analysis import extract_ast_features
from feature_analysis import extract_lexical_features
from feature_analysis import extract_all_features

def main():
    extract_radon_features()
    extract_ast_features()
    extract_lexical_features()
    extract_all_features()
    with open('data/radon_features.json', 'r') as f:
            radondata = json.load(f)
    with open('data/ast_features.json', 'r') as f:
            astdata = json.load(f)
    with open('data/lexical_features.json', 'r') as f:
            lexicaldata = json.load(f)
    with open('data/combined_features.json', 'r') as f:
            combineddata = json.load(f)
    radondf = pd.DataFrame(radondata)
    astdf = pd.DataFrame(astdata)
    lexicaldf = pd.DataFrame(lexicaldata)
    combineddf = pd.DataFrame(combineddata)
    radondf_day3 = radondf[radondf['day'] == 3]
    astdf_day3 = astdf[astdf['day'] == 3]
    lexicaldf_day3 = lexicaldf[lexicaldf['day'] == 3]
    combineddf_day3 = combineddf[combineddf['day'] == 3]

    clustered_radon = run_kmeans(radondf_day3, n_clusters=3)
    clustered_ast = run_kmeans(astdf_day3, n_clusters=3)
    clustered_lexical = run_kmeans(lexicaldf_day3, n_clusters=3)
    clustered_combined = run_kmeans(combineddf_day3, n_clusters=3)
if __name__ == "__main__":
    main()