import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, filename='log_file.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

directory = 'summaries3'
data = []

if not os.path.exists(directory):
    logging.error(f"The directory {directory} does not exist.")
    raise Exception(f"The directory {directory} does not exist.")

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as file:
                content = json.load(file)
                
                entry = {
                    'filename': filename,
                    'rouge_1_recall': content['rouge']['rouge']['rouge_1_recall'],
                    'rouge_1_precision': content['rouge']['rouge']['rouge_1_precision'],
                    'rouge_1_f_score': content['rouge']['rouge']['rouge_1_f_score'],
                    'rouge_2_recall': content['rouge']['rouge']['rouge_2_recall'],
                    'rouge_2_precision': content['rouge']['rouge']['rouge_2_precision'],
                    'rouge_2_f_score': content['rouge']['rouge']['rouge_2_f_score'],
                    'rouge_l_recall': content['rouge']['rouge']['rouge_l_recall'],
                    'rouge_l_precision': content['rouge']['rouge']['rouge_l_precision'],
                    'rouge_l_f_score': content['rouge']['rouge']['rouge_l_f_score']
                }
                data.append(entry)
                logging.info(f"Processed {filename} successfully.")
        
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from file {filename}")
        except Exception as e:
            logging.error(f"An error occurred while processing file {filename}: {str(e)}")

try:
    df = pd.DataFrame(data)
    df.to_csv('summary3_scores.csv', index=False)
    logging.info("Dataframe was successfully saved to summary3_scores.csv")
except Exception as e:
    logging.error(f"Failed to save dataframe to CSV: {str(e)}")
