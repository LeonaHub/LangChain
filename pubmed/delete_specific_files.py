import os

def delete_specific_files(directory, file_names):
    """
    Deletes specific files from a given directory.
    
    :param directory: Path to the directory containing files
    :param file_names: List of filenames to be deleted
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Process each file in the list
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_name}")
            except Exception as e:
                print(f"Error deleting file {file_name}: {str(e)}")
        else:
            print(f"File not found: {file_name}")

if __name__ == "__main__":
    # Path to the directory where the files are stored
    directory_path = './ref_abstracts'
    
    # List of filenames to delete
    file_names_to_delete = [
        "28854875.txt", "18559668.txt", "37512948.txt", "30105209.txt",
        "28106858.txt", "31237568.txt", "35891981.txt", "22615903.txt",
        "36271928.txt", "37771082.txt", "37760634.txt", "34733649.txt",
        "33743585.txt", "33094287.txt", "35367895.txt", "25070835.txt",
        "16309622.txt", "22328786.txt", "10611226.txt", "36139021.txt",
        "36772335.txt", "17407552.txt", "36636026.txt"
    ]
    
    # Call the function to delete files
    delete_specific_files(directory_path, file_names_to_delete)
