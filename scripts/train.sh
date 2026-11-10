#!/bin/bash

python src/scagnn/train.py \
dataset_train='helmholtz_dirichlet' \
dataset_val='helmholtz_dirichlet_test' \
dataset_test='helmholtz_dirichlet_test' \
compare_normalized=True \
predict_normalized=True

python src/scagnn/train.py \
dataset_train='helmholtz_neumann' \
dataset_val='helmholtz_neumann_test' \
dataset_test='helmholtz_neumann_test' \
compare_normalized=False \
predict_normalized=False

python src/scagnn/train.py \
dataset_train='laplace_dirichlet' \
dataset_val='laplace_dirichlet_test' \
dataset_test='laplace_dirichlet_test' \
compare_normalized=True \
predict_normalized=Tru\