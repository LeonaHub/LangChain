import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'Abstracts'
output_file = 'Abstracts_scores.csv'

results = pd.DataFrame(columns=['filename', 'model_type', 'blanc_score'])

logging.info(f'Starting to parse files in directory: {directory}')

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            blanc_score = data.get('blanc', {}).get('blanc', None)
            
            if 'langchain' in filename:
                model_type = 'langchain'
            else:
                model_type = 'openai'
            
            results = results.append({
                'filename': filename,
                'model_type': model_type,
                'blanc_score': blanc_score
            }, ignore_index=True)
            
            logging.info(f'Successfully parsed file: {filename}')
        
        except Exception as e:
            logging.error(f'Error parsing file {filename}: {e}')

results.to_csv(output_file, index=False)

logging.info(f'Parsing complete. Results saved to {output_file}')
