import argparse
import os
from experiments.baseline_experiment import BaselineExperiment
from experiments.concept_aware_experiment import ConceptAwareExperiment

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(script_dir, 'configs', 'base_config.yaml')

    parser = argparse.ArgumentParser(description="Run Concept-Aware LLM Experiments")
    parser.add_argument(
        '--experiment',
        type=str,
        default='baseline',
        choices=['baseline', 'concept_aware'],
        help='Which experiment to run.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='folio',
        choices=['folio', 'truthfulqa'],
        help='Which dataset to use for the experiment.'
    )
    parser.add_argument(
        '--decoding_strategy',
        type=str,
        default='concept_aware',
        choices=['concept_aware', 'entropy_gated'],
        help='Decoding strategy to use for the concept-aware experiment.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help='Path to the base configuration file.'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=20,
        help='Number of examples to run the experiment on.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    print(f"--- Starting {args.dataset.upper()} {args.experiment.replace('_', ' ').title()} Experiment ---")
    
    if args.experiment == 'baseline':
        experiment_runner = BaselineExperiment(config_path=args.config, dataset_name=args.dataset)
    elif args.experiment == 'concept_aware':
        experiment_runner = ConceptAwareExperiment(
            config_path=args.config, 
            dataset_name=args.dataset,
            decoding_strategy=args.decoding_strategy
        )
    else:
        print(f"Unknown experiment: {args.experiment}")
        return

    try:
        scores = experiment_runner.run(num_examples=args.num_examples)
        
        print(f"\n--- {args.dataset.upper()} {args.experiment.replace('_', ' ').title()} Experiment Finished ---")
        print("Scores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.4f}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
