import os
import json
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_article_to_json(article, output_dir, filename):
    """
    Saves the given article as a JSON file in the specified directory with the given filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(article, file, indent=4)
        logging.info(f"Article JSON saved to {file_path}")
    except IOError as e:
        logging.error(f"Failed to write article JSON to {file_path}: {e}")

def main():
    """
    Main function to load dataset and save a specified article as JSON.
    """
    try:
        dataset = load_dataset('pubmed.py', '2024', trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    index = 12345  # Example index
    try:
        article = dataset['train'][index]
        output_dir = 'metadata_json'
        filename = f"article_{index}.json"
        save_article_to_json(article, output_dir, filename)
    except IndexError:
        logging.error(f"Index {index} is out of bounds for the dataset.")
    except KeyError as e:
        logging.error(f"Key error in dataset access: {e}")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
