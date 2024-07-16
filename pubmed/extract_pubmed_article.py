import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_article_to_text(article, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(article.get('Abstract', 'No abstract available'))
        logging.info(f"Article saved to {file_path}")
    except IOError as e:
        logging.error(f"Failed to write article to {file_path}: {e}")

def main():
    try:
        dataset = load_dataset('pubmed.py', '2024', trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    index = 12345  
    try:
        # print(len(dataset['train'])) 36555430
        article = dataset['train'][index]
        output_dir = 'raw_texts'
        filename = f"article_{index}.txt"
        save_article_to_text(article, output_dir, filename)
    except IndexError:
        logging.error(f"Index {index} is out of bounds for the dataset.")
    except KeyError as e:
        logging.error(f"Key error in dataset access: {e}")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
