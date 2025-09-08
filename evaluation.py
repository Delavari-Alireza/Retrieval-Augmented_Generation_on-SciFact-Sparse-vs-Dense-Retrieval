import json
import sys
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


def main():
    """
    This script evaluates the performance of a retrieval system.
    It expects two arguments:
    1. The path to the directory containing the dataset (e.g., 'datasets/scifact').
    2. The path to the results file (e.g., 'results/bm25_results.json').
    """
    if len(sys.argv) != 3:
        print("Usage: python evaluation.py <dataset_path> <results_file_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    results_file_path = sys.argv[2]

    # Load the dataset and qrels
    try:
        _, _, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        sys.exit(1)

    # Load the results from the retrieval system
    try:
        with open(results_file_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file_path}")
        sys.exit(1)

    # Initialize the evaluator
    evaluator = EvaluateRetrieval()

    # Evaluate the results
    print(f"Evaluating results from {results_file_path}...")
    scores = evaluator.evaluate(qrels, results, [10, 100])

    # Print the scores
    print("\nEvaluation Scores:")
    print(json.dumps(scores, indent=4))


if __name__ == "__main__":
    main()
