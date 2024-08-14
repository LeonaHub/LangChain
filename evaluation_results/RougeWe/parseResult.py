import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'summaries3'

results = pd.DataFrame(columns=[
    'filename', 'model_type', 'doc_id', 
    'rouge_we_3_p', 'rouge_we_3_r', 'rouge_we_3_f'
])

logging.info(f'Starting to parse files in directory: {directory}')

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            rouge_we = data.get('rough_we', {})
            rouge_we_3_p = rouge_we.get('rouge_we_3_p', None)
            rouge_we_3_r = rouge_we.get('rouge_we_3_r', None)
            rouge_we_3_f = rouge_we.get('rouge_we_3_f', None)
            
            doc_id = filename.split('_')[1].split('.')[0]
            
            if 'langchain' in filename:
                model_type = 'langchain'
            else:
                model_type = 'openai'
            
            results = results.append({
                'filename': filename,
                'model_type': model_type,
                'doc_id': doc_id,
                'rouge_we_3_p': rouge_we_3_p,
                'rouge_we_3_r': rouge_we_3_r,
                'rouge_we_3_f': rouge_we_3_f
            }, ignore_index=True)
            
            logging.info(f'Successfully parsed file: {filename}')
        
        except Exception as e:
            logging.error(f'Error parsing file {filename}: {e}')

output_file = 'summaries3_scores.csv'
results.to_csv(output_file, index=False)

logging.info(f'Parsing complete. Results saved to {output_file}')
