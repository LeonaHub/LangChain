import os
import json
import logging
from pathlib import Path
from summ_eval.syntactic_metric import SyntacticMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluateSyntacticComplexity:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.syntactic_metric = SyntacticMetric()

    def setup_directories(self):
        self.data_dir = Path(self.config['data_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.syntactic_dir = self.output_dir / "Syntactic"
        self.syntactic_dir.mkdir(parents=True, exist_ok=True)

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def evaluate_summaries(self):
        dir_path = Path(self.config['summaries_dir'] + '/' + self.config[subfolder])
        for file_name in os.listdir(dir_path):
            summary_path = dir_path / file_name
            if summary_path.exists():
                try:
                    summary = self.read_file(summary_path)
                    self.evaluate_and_save_results(summary, dir_path, file_name)
                except Exception as e:
                    logger.error(f"Failed to evaluate syntactic complexity for {summary_path}: {e}")

    def evaluate_and_save_results(self, summary, dir_path, file_name):
        results = self.syntactic_metric.evaluate_batch([summary], [])
        output_file = self.syntactic_dir / f"{dir_path.name}_{file_name}.json"
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump({ "syntactic_complexity": results }, out_file, ensure_ascii=False, indent=4)
        logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    evaluator = EvaluateSyntacticComplexity(config)
    evaluator.evaluate_summaries('openai')
    evaluator.evaluate_summaries('langchain')
