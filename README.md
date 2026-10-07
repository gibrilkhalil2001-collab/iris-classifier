# Iris Classifier (Decision Tree)

## Overview

End-to-end ML example from the Digital Marketing Mastery module. It builds a
decision-tree classifier on the classic Iris dataset using scikit-learn.

## Quick start

```bash
git clone https://github.com/gibrilkhalil2001-collab/iris-classifier.git
cd iris-classifier
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Project structure

- `notebooks/iris_model.ipynb`: step-by-step walk-through notebook
- `src/train.py`: reproducible command-line script
- `tests/test_train.py`: basic test (accuracy of at least 0.9)
- `outputs/`: generated confusion matrix figure
- `requirements.txt`: Python packages needed
- `LICENSE`: MIT license

## Results

The decision tree reaches about 100% accuracy on the 20% test split
(random-state 42). Petal length is the most important feature.

## License

MIT
