## Evaluation

This folder contains files for evaluating the models. Running `eval_gips.py` creates a json file (`eval.json`) that contains:
- basic information about the model,
- the total loss(es) for the test dataset,
- the normalized loss(es) for the test dataset (indicated by `n_`),
- the total distance for all predictions to the actual location,
- and a list of all distances

The metrics (distance) are computed using `metrics.py`.

`visualize_eval.ipynb` visualizes the evaulation by plotting a histogram over prediction distances form the test set.