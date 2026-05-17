import pandas as pd
import ast
import re
import os

class SaltDataIngestor:
    """
    Parses the SALT-NLP 'all.csv' files
    into a clean format for benchmarking.
    """
    _cached_scenarios = None

    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(base_dir, "data", "all.csv")
        else:
            self.data_path = data_path

    def get_scenarios(self):
        if SaltDataIngestor._cached_scenarios is not None:
            return SaltDataIngestor._cached_scenarios

        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found.")
            return []

        print(f"--- Loading Scenario Cache from {self.data_path} ---", flush=True)
        df = pd.read_csv(self.data_path)
        
        formatted_scenarios = []
        for idx, row in df.iterrows():
            base_text = str(row.get('base_text', ''))
            human_clue_word = str(row.get('output', ''))
            
            match_black = re.search(r"black:\s*(\[.*?\])", base_text)
            match_tan = re.search(r"tan:\s*(\[.*?\])", base_text)
            match_targets = re.search(r"targets:\s*(\[.*?\])", base_text)
            
            try:
                black = ast.literal_eval(match_black.group(1)) if match_black else []
                tan = ast.literal_eval(match_tan.group(1)) if match_tan else []
                targets = ast.literal_eval(match_targets.group(1)) if match_targets else []
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                continue
                
            scenario = {
                "team": targets,
                # SALT does not separate opponent from assassin — 'black' is all
                # words to avoid.  Expose it raw; callers decide how to use it.
                "opponent": [],
                "neutral": tan,
                "assassin": None,
                "human_clues": [{"word": human_clue_word, "targets": targets}],
                "black": black,
            }
            formatted_scenarios.append(scenario)
            
        SaltDataIngestor._cached_scenarios = formatted_scenarios
        return formatted_scenarios

if __name__ == "__main__":
    ingestor = SaltDataIngestor("data/all.csv")
    scenarios = ingestor.get_scenarios()
    print(f"Parsed {len(scenarios)} scenarios from SALT-NLP.")
