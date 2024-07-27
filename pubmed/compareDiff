import os

def compare_directories(dir1, dir2):
    # Get a list of file names from both directories
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    # Find files unique to each directory
    unique_to_dir1 = files_dir1 - files_dir2
    unique_to_dir2 = files_dir2 - files_dir1

    # Print the results
    print(f"Files only in {dir1}:")
    for file in unique_to_dir1:
        print(file)

    print(f"\nFiles only in {dir2}:")
    for file in unique_to_dir2:
        print(file)

dir1 = 'raw_texts'
dir2 = 'ref_abstracts'
compare_directories(dir1, dir2)
