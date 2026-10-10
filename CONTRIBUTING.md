# Contributing to DARKGUARD

DARKGUARD is a baseline NLP/ML prototype prepared for a coding event. The open GitHub Issues are the task list; each issue has an `easy`, `medium`, or `hard` difficulty label.

## Workflow

1. Read the open Issues and choose one that matches your experience.
2. Comment on the issue before starting so work does not overlap.
3. Fork or clone the repository and create a branch for your change.
4. Reproduce the current baseline before changing the model.
5. Keep the change focused on the selected issue.
6. Run the available tests and check the app locally.
7. Open a Pull Request that links the issue (for example, `Closes #2`).

## ML evaluation

For model changes, report accuracy, precision, recall, F1-score, and a confusion matrix where applicable. Describe the data split and random seed, include representative errors, and compare against the existing baseline.

Keep the held-out test set separate from training and tuning. Do not claim improvement based only on training results or a small hand-picked sample. For category or multi-label work, state clearly which labels the dataset actually supports.

## General guidelines

- Explain what changed, why, and how it was tested.
- Keep code and explanations understandable to other students.
- Do not commit secrets, virtual environments, or generated cache files.
- A prediction is not proof that a website intentionally deceives users.

Difficulty labels describe expected scope, not guaranteed completion time.