import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'summaries3'
output_file = 'summaries3_scores.csv'

results = pd.DataFrame(columns=['filename', 'model_type', 'doc_id', 'summaqa_avg_prob', 'summaqa_avg_fscore'])

logging.info(f'Starting to parse files in directory: {directory}')

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            summaqa = data.get('summa_qa', {})
            avg_prob = summaqa.get('summaqa_avg_prob', None)
            avg_fscore = summaqa.get('summaqa_avg_fscore', None)
            
            doc_id = filename.split('_')[1].split('.')[0]
            if 'langchain' in filename:
                model_type = 'langchain'
            else:
                model_type = 'openai'
            
            results = results.append({
                'filename': filename,
                'model_type': model_type,
                'doc_id': doc_id,
                'summaqa_avg_prob': avg_prob,
                'summaqa_avg_fscore': avg_fscore
            }, ignore_index=True)
            
            logging.info(f'Successfully parsed file: {filename}')
        
        except Exception as e:
            logging.error(f'Error parsing file {filename}: {e}')

results.to_csv(output_file, index=False)

logging.info(f'Parsing complete. Results saved to {output_file}')
