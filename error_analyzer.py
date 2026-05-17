import pandas as pd

class ErrorAnalyzer:
    def __init__(self):
        self.logs = []

    def log_failure(self, scenario, clue, failure_type):
        self.logs.append({
            "clue": clue,
            "failure_type": failure_type,
            "team_size": len(scenario['team']),
            "assassin": scenario['assassin']
        })

    def generate_report(self):
        df = pd.DataFrame(self.logs)
        print("\\n--- Failure Distribution ---")
        if not df.empty:
            print(df['failure_type'].value_counts(normalize=True) * 100)
        else:
            print("No failures logged.")

# Example Usage:
# if top_clue_dist_to_assassin < 0.3:
#    analyzer.log_failure(scenario, top_clue, "ASSASSIN_RISK")
