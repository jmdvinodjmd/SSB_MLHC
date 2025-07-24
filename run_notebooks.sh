#!/bin/bash

# List of notebook files to run
notebooks=("2a_TNet.ipynb" "2b_MT.ipynb" "2c_AdvNet.ipynb" "2d_imputation.ipynb" "2e_IPW.ipynb" "2f_KMM_KLIEP.ipynb")

# Run each notebook
for notebook in "${notebooks[@]}"
do
    echo '-----------------------------------------------------------------'
    echo $notebook
    jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=-1 "$notebook"
done
