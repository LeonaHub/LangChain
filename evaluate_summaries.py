import os
import json
import logging
from pathlib import Path
from summ_eval import (
    BertScoreMetric, BlancMetric, BleuMetric, ChrfppMetric, CiderMetric,
    DataStatsMetric, MeteorMetric, MoverScoreMetric, RougeMetric, RougeWeMetric,
    S3Metric, SentenceMoversMetric, SummaQAMetric, SupertMetric, SyntacticMetric
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluateSummaries:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.initialize_metrics()

    def setup_directories(self):
        self.data_dir = Path(self.config['data_dir'])
        self.references_dir = Path(self.config['references_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dirs = [Path(self.config[d]) for d in self.config if 'summaries_dir' in d]

    def initialize_metrics(self):
        self.metrics = {
            "bert_score": BertScoreMetric(),
            "blanc": BlancMetric(),
            "bleu": BleuMetric(),
            "chrfpp": ChrfppMetric(),
            "cider": CiderMetric(),
            "data_stats": DataStatsMetric(),
            "meteor": MeteorMetric(),
            "mover_score": MoverScoreMetric(),
            "rouge": RougeMetric(),
            "rouge_we": RougeWeMetric(),
            "s3": S3Metric(),
            "sentence_movers": SentenceMoversMetric(),
            "summa_qa": SummaQAMetric(),
            "supert": SupertMetric(),
            "syntactic": SyntacticMetric()
        }

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
                        logger.error(f"Error processing {file_name}: {e}")

    def evaluate_and_save_results(self, summary, reference, dir_path, file_name):
        results = {metric_name: metric.evaluate_batch([summary], [reference]) for metric_name, metric in self.metrics.items()}
        output_file = self.output_dir / f"{dir_path.name}_{file_name}.json"
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump(results, out_file, ensure_ascii=False, indent=4)
        logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    evaluator = EvaluateSummaries(config)
    evaluator.evaluate_summaries()
