import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'summaries3'

results = pd.DataFrame(columns=['filename', 'model_type', 'bert_score_precision', 'bert_score_recall', 'bert_score_f1'])

logging.info(f'Starting to parse files in directory: {directory}')

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            bert_score = data.get('bert_score', {})
            precision = bert_score.get('bert_score_precision', None)
            recall = bert_score.get('bert_score_recall', None)
            f1_score = bert_score.get('bert_score_f1', None)
            
            if 'langchain' in filename:
                model_type = 'langchain'
            else:
                model_type = 'openai'
            
            results = results.append({
                'filename': filename,
                'model_type': model_type,
                'bert_score_precision': precision,
                'bert_score_recall': recall,
                'bert_score_f1': f1_score
            }, ignore_index=True)
            
            logging.info(f'Successfully parsed file: {filename}')
        
        except Exception as e:
            logging.error(f'Error parsing file {filename}: {e}')

output_file = 'summary3_scores.csv'
results.to_csv(output_file, index=False)

logging.info(f'Parsing complete. Results saved to {output_file}')
