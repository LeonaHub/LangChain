import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def get_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def summarize_text(text, api_key):
    model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    try:
        options = {
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.5,
            "frequency_penalty": 1.5,
            "presence_penalty": 0
        }
        response = model.invoke([
            SystemMessage(content="Summarize this text:"),
            HumanMessage(content=text)
        ], **options)
        print(response.content)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def main():
    api_key = get_api_key()

    text = read_text_from_file("text.txt")
    summarize_text(text, api_key)

if __name__ == "__main__":
    main()
