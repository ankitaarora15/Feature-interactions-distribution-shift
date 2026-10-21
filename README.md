# Feature Interactions Under Distribution Shift

## Overview

This project investigates how class-preserving reweighting of selected feature-defined subgroups changes evaluation metrics for three classifiers on the UCI Adult Census Income dataset. The current experiments use Logistic Regression, Decision Tree, and Random Forest.

**Important interpretation:** these experiments reweight fixed test records during evaluation. The models are not retrained under a new distribution, and no new test population is collected. Results therefore describe sensitivity of weighted evaluation metrics to the chosen subgroup weighting scheme; they do not establish performance on a genuinely shifted population.

## Research question

How do individual and combined changes in input-feature distributions affect the predictive performance of machine-learning models?

In this initial study, this question is investigated using controlled, class-preserving subgroup reweighting.

## Repository contents

- `01_dataset_inspection.ipynb` — loads and explores the dataset only.
- `02_project_report.ipynb` — report, results, discussion, and limitations.
- `03_complete_research_workflow.ipynb` — model training, baseline evaluation, reweighting experiments, result CSV generation, and Education-strength graph generation.
- `consolidated_shift_results.csv` — baseline and multiplier-3 results for single and combined subgroup conditions.
- `education_strength_results.csv` — Education subgroup reweighting results for multipliers 1, 2, 3, and 5.
- `model_performance_graph.png` — grouped comparison of balanced accuracy across all conditions and models.
- `education_strength_graph.png` — balanced-accuracy sensitivity to Education subgroup weight multiplier.

## Dataset setup

Download the Adult dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/2/adult

Place the original files `adult.data` and `adult.test` in the same folder as the notebooks. Do not upload these dataset files to GitHub unless you have checked the dataset's distribution terms and repository policy.

## Run the project

1. Install Python 3.10+ and the required packages:

   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

2. Open Jupyter in the project folder that contains `adult.data` and `adult.test`.
3. Run `01_dataset_inspection.ipynb` to inspect the data.
4. Run `03_complete_research_workflow.ipynb` from top to bottom. It regenerates `consolidated_shift_results.csv`, `education_strength_results.csv`, and `education_strength_graph.png`.
5. Regenerate the main comparison graph from the consolidated CSV by running `python make_model_performance_graph.py` in the project folder.
6. Compare the generated CSV values with the report in `02_project_report.ipynb`.

## Experimental conditions

The subgroup masks are defined as:

- Age: age >= 40
- Hours: hours per week > 45
- Education: education number > 12
- Combined conditions: intersections of the above masks

In the main comparison, selected records receive multiplier 3 before weights are normalized within each income class. In the Education-strength sensitivity analysis, multipliers 1, 2, 3, and 5 are tested.

## Limitations

- One dataset and one fixed train/test split are used.
- The subgroup cutoffs are operational choices and should be tested for sensitivity.
- No confidence intervals or repeated-split uncertainty estimates are included.
- Reweighting is not equivalent to evaluating on a newly collected population or retraining under distribution shift.
- The Adult dataset is historical and has representativeness and fairness limitations.
- No novelty claim is made; a literature review is needed before any novelty assessment.

## Main observed pattern

In this experiment, Education subgroup reweighting produced the largest balanced-accuracy increase among the three individual feature conditions for all three models. The direction and magnitude of changes depend on the model and weighting scheme. An increase in balanced accuracy should not be interpreted as a general improvement in the model or proof of better real-world performance.
