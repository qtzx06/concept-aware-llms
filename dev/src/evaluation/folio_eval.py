import re
from collections import Counter

class FolioEvaluator:
    def __init__(self):
        pass

    def _normalize_answer(self, text):
        """Lowercases, removes punctuation, and strips whitespace."""
        text = text.lower().strip()
        # Keep only the first word (e.g., "true." -> "true")
        return re.split(r'[\s\.\,]', text)[0]

    def evaluate(self, predictions, ground_truths):
        """
        Calculates accuracy for the FOLIO task.
        :param predictions: A list of generated answers (strings) from the model.
        :param ground_truths: A list of dictionaries from the dataset, each with a 'label' key.
        :return: A dictionary with the accuracy score.
        """
        correct = 0
        total = len(predictions)

        if total == 0:
            return {"accuracy": 0}

        for i in range(total):
            predicted_label_raw = predictions[i]
            true_label = ground_truths[i]['label'].lower()
            if true_label == 'uncertain':
                true_label = 'unknown'
            
            predicted_label = self._normalize_answer(predicted_label_raw)

            if predicted_label == true_label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

if __name__ == '__main__':
    # Example Usage
    evaluator = FolioEvaluator()

    example_predictions = [
        "True.",
        "False",
        "Unknown because it is not stated.",
        "True"
    ]

    example_ground_truths = [
        {"label": "True"},
        {"label": "False"},
        {"label": "Unknown"},
        {"label": "False"} # Model got this one wrong
    ]

    results = evaluator.evaluate(example_predictions, example_ground_truths)
    print("Evaluation results:", results)
    # Expected output:
    # Accuracy: 3/4 = 0.75
