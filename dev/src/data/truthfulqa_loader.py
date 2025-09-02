import os
import pandas as pd

class TruthfulQALoader:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        self.data_path = os.path.join(project_root, "TruthfulQA", "TruthfulQA.csv")
        self._ensure_data_exists()
        print("Loading local TruthfulQA data...")

    def _ensure_data_exists(self):
        if not os.path.exists(self.data_path):
            print("TruthfulQA data not found. Please run:")
            print("git clone https://github.com/sylinrl/TruthfulQA.git")
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

    def load_data(self, num_examples=None):
        """
        Loads data from the local CSV file.
        """
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error reading CSV file at {self.data_path}: {e}")
            raise
        
        if num_examples:
            df = df.head(num_examples)
            
        return df.to_dict('records')

    def format_prompt(self, example_data):
        # Using the standard Question-Answer prompt format
        question = example_data['Question']
        prompt = f"Q: {question}\nA:"
        return prompt

if __name__ == '__main__':
    try:
        loader = TruthfulQALoader() 
        sample_data = loader.load_data(num_examples=3)
        if sample_data:
            print(f"\nLoaded {len(sample_data)} examples.")
            for item in sample_data:
                print("---")
                print("Question:", item['Question'])
                print("Best Answer:", item['Best Answer'])
                print("\nFormatted prompt:\n", loader.format_prompt(item))
    except Exception as e:
        print(f"Failed to run example: {e}")
