In the paper by Biel et al. (2023), they proposed 35 specific features to identify human behavorial traits in coding

These were the lexical features they proposed:
1. Length of lines (average, 80th percentile)
2. Length of variables (average, 80th percentile)
3. Length of methods in lines (average, maximum, 80th percentile)
4. Length of comments in characters (average, maximum, 80th percentile)
5. Length of comments in lines ratio to code length
6. Length of commented out code ratio
7. Number of lines with more than one instruction ratio
8. Number of consecutive lines with aligned characters - 4 features
9. Number of white characters ratio
10. Number of occurrences of -1 (special value) ratio
11. Ratio of English words to other languages in names
12. Preserving naming convention (PascalCase, snake_case)
13. Consistency in curly bracket placement (next line vs end of line) - 2 features
14. Consistent application of curly brackets around one-line branches
15. Level of indentation correctness

These are the other features they proposed 
Syntactic Features (2):
16. Degree of exploitation of language syntax (various for loops, lambdas)
17. Depth of references to fields and methods (max, 80th percentile)
Semantic Features (7):
18. Number of methods in a class (average, maximum, 80th percentile)
19. Number of used switch instructions ratio
20. Number of separated logic blocks within methods ratio
21. Number of code duplications ratio
22. Maximum nesting depth of instructions


Consistency in using curly brackets around one-line branches of code is implemented in two variants so it gives rise to two features. 
Number of consecutive lines with aligned characters represents four features, as it is computed separately for four groups of characters.

In the code I will comment when a certain feature is computed