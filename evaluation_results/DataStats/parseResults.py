import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'summaries3'

results = pd.DataFrame(columns=[
    'filename', 'model_type', 'doc_id', 'coverage', 'density', 'compression', 
    'percentage_novel_1-gram', 'percentage_novel_2-gram'
])

logging.info(f'Starting to parse files in directory: {directory}')

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            data_stats = data.get('DataStats', {})
            coverage = data_stats.get('coverage', None)
            density = data_stats.get('density', None)
            compression = data_stats.get('compression', None)
            percentage_novel_1 = data_stats.get('percentage_novel_1-gram', None)
            percentage_novel_2 = data_stats.get('percentage_novel_2-gram', None)
            
            doc_id = filename.split('_')[1].split('.')[0]
            
            if 'langchain' in filename:
                model_type = 'langchain'
            else:
                model_type = 'openai'
            
            results = results.append({
                'filename': filename,
                'model_type': model_type,
                'doc_id': doc_id,
                'coverage': coverage,
                'density': density,
                'compression': compression,
                'percentage_novel_1-gram': percentage_novel_1,
                'percentage_novel_2-gram': percentage_novel_2
            }, ignore_index=True)
            
            logging.info(f'Successfully parsed file: {filename}')
        
        except Exception as e:
            logging.error(f'Error parsing file {filename}: {e}')

output_file = 'summary3_scores.csv'
results.to_csv(output_file, index=False)

logging.info(f'Parsing complete. Results saved to {output_file}')
