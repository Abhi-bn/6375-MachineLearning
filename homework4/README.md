# Multi-Class-Bayesian
## Running Instruction
1. Use MultiClassCompare.ipynb to run the code to get prediction results
2. Change `dname` to change the dataset {Sample_1_MLC_2022, Sample_2_MLC_2022, Sample_3_MLC_2022, Sample_4_MLC_2022}

## Feature Generations
1. Use MultiClassHiddenAssignments.ipynb file to generate features
2. Change `dname` to change the dataset {Sample_1_MLC_2022, Sample_2_MLC_2022, Sample_3_MLC_2022, Sample_4_MLC_2022}


## Note:
1. content/MLC contains the .data and .uai files along with .order and .new_features
    1. .order represents order of `variable elimination`
    2. .new_features file contains new augmented `hidden Assignments` for all the samples
2. Generating hidden features takes longer time, so adding the generated features to the git repository.
3. Added results by computing score for all test points, recommend testing it with 10 test points to save time.

## Report
1. Please fine the report ML-Assignment4.pdf