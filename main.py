import shutil
from pathlib import Path
from src.classifier import MessageClassifier
from src.file_handler import load_json, save_json, read_file

# --- Configuration ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# --- Base Paths ---
ROOT_DIR = Path(__file__).parent
PROMPTS_DIR = ROOT_DIR / "prompts"
EXAMPLES_DIR = ROOT_DIR / "examples"



def run_experiment(exp_dir: Path):
    """Runs a full classification experiment for a given number."""

    if not exp_dir.exists():
        print(f"Error: Experiment directory {exp_dir} not found.")
        return

    messages_path = exp_dir / "messages.json"
    shot_examples_path = exp_dir / "shot_examples.json"
    results_dir = exp_dir / "results"

    # Clear previous results
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir()

    # Load data
    data = load_json(messages_path)
    messages = data.get("messages", [])
    categories = data.get("categories", [])
    shot_examples = load_json(shot_examples_path)

    if not messages or not categories:
        print(f"Error: 'messages' or 'categories' not found in {messages_path}.")
        return

    classifier = MessageClassifier(base_url=LM_STUDIO_BASE_URL)
    scenarios = ["zero_shot", "one_shot", "few_shot"]

    for scenario in scenarios:
        print(f"\n--- Running Scenario: {scenario.upper()} ---")
        # Load prompts for the current scenario
        # system_prompt = read_file(PROMPTS_DIR / scenario / "system.txt").format(categories=', '.join(categories))
        system_prompt = read_file(PROMPTS_DIR / "system.txt").format(categories=', '.join(categories))
        # user_prompt_template = read_file(PROMPTS_DIR / scenario / "user.txt")

        results = {}
        for msg in messages:
            
            message_list = [
                {"role": "system", "content": system_prompt}
            ]

            if scenario == 'zero_shot':
                message_list.append({"role": "user", "content": msg})

            else:
                for example in shot_examples['examples']:
                    message_list.append({"role": "user", "content": example['message']})
                    message_list.append({"role": "assistant", "content": example['categories']})

                    if scenario == 'one_shot':
                        break

                message_list.append({"role": "user", "content": msg})

            predicted_categories = classifier.classify_message(message_list)
            results[msg] = predicted_categories
            print(f'Message: "{msg}" -> Categories: {predicted_categories}')

        # Save results for the scenario
        results_path = results_dir / f"{scenario}_results.json"
        save_json(results, results_path)
        print(f"Successfully saved results to {results_path}")

def main():
    """Main function to find and run all experiments."""
    for exp_dir in EXAMPLES_DIR.iterdir():
        if exp_dir.is_dir():
            print(f"\n{'='*20} RUNNING EXPERIMENT: {exp_dir.name} {'='*20}")
            run_experiment(exp_dir)

if __name__ == "__main__":
    main()
