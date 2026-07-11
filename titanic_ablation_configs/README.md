# Titanic Logistic-Regression Ablation Configs

All experiments hold these choices constant:

- model: logistic regression
- C: 1.0
- random seed: 42
- evaluation: shuffled 5-fold StratifiedKFold
- metric: accuracy
- hyperparameter search: disabled
- feature fitting: performed inside each training fold

## Phase 1 — Forward feature ladder

Run in this order:

1. `a00_core_baseline.yaml`
2. `a01_title_age.yaml`
3. `a02_family.yaml`
4. `a03_ticket_fare.yaml`
5. `a04_cabin.yaml`
6. `a05_group_frequency.yaml`
7. `a06_full_interactions.yaml`
8. `a07_full_with_bins.yaml`

For adjacent experiments, calculate:

`incremental_delta = current_mean_cv_accuracy - previous_mean_cv_accuracy`

This reveals the marginal effect of adding each feature group, although the result can
depend on the order in which groups are added.

## Phase 2 — Leave-one-group-out ablations

Use `a06_full_interactions.yaml` as the reference full system.

- `a10_full_minus_family.yaml`
- `a11_full_minus_ticket.yaml`
- `a12_full_minus_cabin.yaml`
- `a13_full_minus_group_frequency.yaml`
- `a14_full_minus_interactions.yaml`
- `a15_full_minus_fare_derived.yaml`

For each experiment, calculate:

`ablation_delta = full_mean_cv_accuracy - ablated_mean_cv_accuracy`

A positive value means removing the group hurt performance. A zero or negative value means
the group was unnecessary or harmful under this model.

## Selection rule

Do not select features from mean accuracy alone. Prefer a feature group when:

1. it improves mean CV accuracy,
2. the improvement is not caused by one unusually strong fold,
3. fold-to-fold standard deviation does not materially worsen,
4. the effect is supported by both the forward ladder and leave-one-out ablation.

After selecting the feature set, freeze it. Only then compare models and tune
hyperparameters.
