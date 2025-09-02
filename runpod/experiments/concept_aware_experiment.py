import os
import yaml
import json
from datetime import datetime

# Data Loaders
from src.data.folio_loader import FolioLoader
from src.data.truthfulqa_loader import TruthfulQALoader

# Models
from src.models.concept_decoder import ConceptDecoder

# Evaluators
from src.evaluation.folio_eval import FolioEvaluator
from src.evaluation.truthfulqa_eval import TruthfulQAEvaluator

class ConceptAwareExperiment:
    def __init__(self, config_path, dataset_name, decoding_strategy='concept_aware'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_name = dataset_name
        self.decoding_strategy = decoding_strategy
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        if self.dataset_name == 'folio':
            self.data_loader = FolioLoader()
            self.evaluator = FolioEvaluator()
        elif self.dataset_name == 'truthfulqa':
            self.data_loader = TruthfulQALoader()
            self.evaluator = TruthfulQAEvaluator()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def run(self, num_examples=20):
        print(f"Loading {self.dataset_name} data...")
        if self.dataset_name == 'folio':
            test_data = self.data_loader.load_data(split="validation", num_examples=num_examples)
        else:
            test_data = self.data_loader.load_data(num_examples=num_examples)
            
        prompts = [self.data_loader.format_prompt(q) for q in test_data]

        print("Loading model for Concept-Aware Decoding...")
        decoder = ConceptDecoder(
            model_name=self.config['model_name'],
            device=self.config.get('device', 'auto'),
            decoding_strategy=self.decoding_strategy,
            entropy_threshold=self.config.get('concept_processing', {}).get('entropy_threshold', 2.5),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 0.95)
        )

        print(f"Generating concept-aware answers for {len(prompts)} examples...")
        max_new_tokens = 50 if self.dataset_name == 'truthfulqa' else 5
        results = decoder.generate_answers(prompts, max_new_tokens=max_new_tokens)
        generated_answers = [r['prediction'] for r in results]

        print("Evaluating results...")
        scores = self.evaluator.evaluate(generated_answers, test_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{self.dataset_name}_concept_aware_results_{timestamp}.json"
        results_path = os.path.join(self.results_dir, results_filename)

        results_data = {
            "config": self.config,
            "scores": scores,
            "num_examples": len(test_data),
            "predictions": [
                {
                    "prompt": prompts[i],
                    "prediction": results[i]['prediction'],
                    "concepts": results[i]['concepts'],
                    "generation_log": results[i].get('generation_log', []), # Use .get for safety
                    "ground_truth_data": test_data[i]
                } for i in range(len(test_data))
            ]
        }

        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"Results saved to {results_path}")
        return scores
