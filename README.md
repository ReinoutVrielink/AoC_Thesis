# Detecting Coding Personas in Advent of Code Solutions

Bachelor thesis by Reinout Vrielink
Detects coding personas in Python Advent of Code solutions usinghand-crafted features (Radon, AST, lexical) and GraphCodeBERT
embeddings, and relates them to solution efficiency.

To run this code, make sure you first do the following:

    pip install pandas numpy scikit-learn matplotlib seaborn umap-learn hdbscan scipy torch transformers radon

## Data

solutions.json — raw Reddit dataset (compiled by Niek Biesterbos)
combined_features.json - solutions with all of their feature values put together in vectors
final_data_v1.json — external timed solutions (provided by Martijn Huls)


the following files should be ran
python main.py              # feature extraction, clustering, efficiency analysis
python inspection_script.py # robustness checks + RQ1 comparison

## Files

preprocessing.py — cleans raw data, drops invalid/Python 2 code, splits part1/part2
feature_analysis.py — extracts Radon, AST, and lexical features
GraphCodeBert.py — generates and loads GraphCodeBERT embeddings
clustering.py — K-means and per-puzzle HDBSCAN clustering
persona_aggregation.py — Ward aggregation of local clusters into personas
clusteringanalysis.py — within-puzzle z-scoring, robustness checks, clustering comparison
efficiency_analysis.py — matches external runtimes to clusters
inspection_script.py — used for inspecting the data during the whole project, and also runs the robustness + agreement analysis
run_code.py — runtime measurement (by Martijn Huls. this file is only included for reference, not ran)

Generative AI (Gemini, Claude) was used for figure styling and for creating the function that is validating the preprocessed data. All feature extraction, clustering, analysis, and interpretation are my own or the source is referenced
