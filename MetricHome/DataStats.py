import os
import json
import logging
from pathlib import Path
from summ_eval.data_stats_metric import DataStatsMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluateSummaries:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.data_stats = DataStatsMetric()

    def setup_directories(self):
        self.data_dir = Path(self.config['data_dir'])
        self.references_dir = Path(self.config['data_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.data_stats_dir = self.output_dir / "DataStats"
        self.data_stats_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dirs = [Path(self.config[d]) for d in self.config if 'summaries_dir' in d]

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def evaluate_summaries(self):
        for dir_path in self.summary_dirs:
            for file_name in os.listdir(dir_path):
                summary_path = dir_path / file_name
                reference_path = self.references_dir / file_name

                if reference_path.exists() and summary_path.exists():
                    try:
                        summary = self.read_file(summary_path)
                        reference = self.read_file(reference_path)
                        self.evaluate_and_save_results(summary, reference, dir_path, file_name)
                    except Exception as e:
                        logger.error(f"Failed to evaluate DataStats for {summary_path} against {reference_path}: {e}")

    def evaluate_and_save_results(self, summary, reference, dir_path, file_name):
        results = self.data_stats.evaluate_batch([summary], [reference])
        output_file = self.data_stats_dir / f"{dir_path.name}_{file_name}.json"
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump({ "DataStats": results }, out_file, ensure_ascii=False, indent=4)
        logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    evaluator = EvaluateSummaries(config)
    evaluator.evaluate_summaries()
