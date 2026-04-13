import csv
import os
from datetime import datetime

class CSVLogger:
    """
    Logger to save FL training results into a CSV file within a dedicated results folder.
    """
    def __init__(self, config):
        self.config = config
        self.results_dir = "results"
        self._ensure_results_dir()
        self.filename = self._generate_filename()
        self.filepath = os.path.join(self.results_dir, self.filename)
        self._initialize_csv()
        
    def _ensure_results_dir(self):
        """Ensures the results directory exists."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _generate_filename(self):
        """
        Generates a unique filename based on configuration and timestamp.
        Pattern: results_{dataset}_{partition}_{clients}_{rounds}_{timestamp}.csv
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"results_{self.config.dataset_name}_"
            f"{self.config.partition_type}_"
            f"{self.config.num_clients}_"
            f"{self.config.rounds}_"
            f"{timestamp}.csv"
        )
        return filename

    def _initialize_csv(self):
        """Creates the CSV file and writes the header."""
        with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Accuracy (%)"])

    def log_round(self, round_num, accuracy):
        """Appends a round result to the CSV file."""
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, f"{accuracy:.4f}"])
