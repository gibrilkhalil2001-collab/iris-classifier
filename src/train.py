"""Train a decision tree on the Iris dataset.

Run from the project root:
    python src/train.py --test-size 0.2 --random-state 42
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lets the script save images without opening a window
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def run(test_size=0.2, random_state=42, output_dir="outputs"):
    """Train, evaluate, save the confusion matrix figure, return accuracy."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(matrix, display_labels=iris.target_names)
    display.plot(cmap="Blues")
    plt.title("Iris Decision Tree - Confusion Matrix")
    plt.savefig(out / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train an Iris decision tree.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    accuracy = run(args.test_size, args.random_state)
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix saved to outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()