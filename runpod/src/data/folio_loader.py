import os
import jsonlines
import pandas as pd

class FolioLoader:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        self.data_dir = os.path.join(project_root, "FOLIO/data/v0.0")
        self.train_path = os.path.join(self.data_dir, "folio-train.jsonl")
        self.validation_path = os.path.join(self.data_dir, "folio-validation.jsonl")
        self._ensure_data_exists()

    def _ensure_data_exists(self):
        if not os.path.exists(self.train_path) or not os.path.exists(self.validation_path):
            print("FOLIO dataset not found. Please run:")
            print("git clone https://github.com/Yale-LILY/FOLIO.git")
            raise FileNotFoundError("FOLIO dataset not found.")
        else:
            print("FOLIO dataset found.")

    def load_data(self, split="train", num_examples=None):
        path = self.train_path if split == "train" else self.validation_path
        
        examples = []
        try:
            with jsonlines.open(path) as reader:
                for i, obj in enumerate(reader):
                    if num_examples and i >= num_examples:
                        break
                    examples.append(obj)
        except IOError as e:
            print(f"Error reading data file: {e}")
            return []
        return examples

    def format_prompt(self, example_data):
        premises = "\n- ".join(example_data['premises'])
        conclusion = example_data['conclusion']
        
        prompt = (
            "Given the following premises:\n"
            f"- {premises}\n\n"
            "Is the following conclusion True, False, or Unknown?\n"
            f"Conclusion: {conclusion}\n\n"
            "Answer:"
        )
        return prompt

if __name__ == '__main__':
    # Example usage
    try:
        loader = FolioLoader()
        sample_data = loader.load_data(num_examples=3)
        if sample_data:
            print(f"Loaded {len(sample_data)} examples.")
            for item in sample_data:
                print("---")
                print("Original data:", item)
                print("\nFormatted prompt:\n", loader.format_prompt(item))
        else:
            print("Failed to load data.")
    except FileNotFoundError as e:
        print(e)
