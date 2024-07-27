import logging
import random
import os
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_random_indices(limit, num):
    """Generate a list of random indices within a given range."""
    return [random.randint(0, limit - 1) for _ in range(num)]

def create_empty_files(pmids):
    """Create empty text files named after each PMID in specified directories."""
    directories = ['texts', 'textsAbstract']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory {directory}")

        for pmid in pmids:
            file_path = os.path.join(directory, f"{pmid}.txt")
            with open(file_path, 'w') as file:
                pass  # Create an empty file
            logging.info(f"Created empty file {file_path}")

def main():
    """ Main function to load dataset and manage PMIDs. """
    try:
        dataset = load_dataset('pubmed.py', '2024', trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    num_articles = 500
    max_index = len(dataset['train']) - 1
    indices = generate_random_indices(max_index, num_articles)
    print(indices)

    pmids = []
    for index in indices:
        try:
            article = dataset['train'][index]
            pmid = article['MedlineCitation']['PMID']
            pmids.append(pmid)
        except IndexError:
            logging.error(f"Index {index} is out of bounds for the dataset.")
        except KeyError as e:
            logging.error(f"Key error in dataset access: {e}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")

    create_empty_files(pmids)

if __name__ == "__main__":
    main()
