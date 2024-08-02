import sys
from transformers import GPT2Tokenizer

def count_tokens(file_path):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    # Encode the text and count tokens
    tokens = tokenizer.encode(text, return_tensors=None)
    return len(tokens)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_tokens.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    token_count = count_tokens(file_path)
    if token_count is not None:
        print(f"Total tokens in file: {token_count}")
