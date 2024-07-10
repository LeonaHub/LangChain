import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryProcessor:
    def __init__(self, config):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.data_dir = Path(config['data_dir'])
        self.summaries_dir_openai = Path(config['summaries_dir_openai'])
        self.summaries_dir_langchain = Path(config['summaries_dir_langchain'])
        self.references_dir = Path(config['references_dir'])
        self.summary_options = config['summary_options']
        self.setup_directories()

    def setup_directories(self):
        self.summaries_dir_openai.mkdir(parents=True, exist_ok=True)
        self.summaries_dir_langchain.mkdir(parents=True, exist_ok=True)

    def read_text_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def write_text_to_file(self, text, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            raise

    def generate_summary(self, model_type, text):
        try:
            if model_type == "openai":
                return self.generate_with_openai(text)
            elif model_type == "langchain-openai":
                return self.generate_with_langchain(text)
        except Exception as e:
            logger.error(f"Error generating summary with {model_type}: {e}")
            return None 

    def generate_with_openai(self, text):
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Summarize this text:"}, {"role": "user", "content": text}],
            **self.summary_options
        )
        return response.choices[0].message.content

    def generate_with_langchain(self, text):
        model = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")
        options = {
            "temperature": 0.1, 
            "max_tokens": 512, 
            "top_p": 0.5, 
            "frequency_penalty": 1.5, 
            "presence_penalty": 0
        }
        response = model.invoke([SystemMessage(content="Summarize this text:"), HumanMessage(content=text)], **options)
        return response.content

    def process_texts(self):
        for text_file in self.data_dir.iterdir():
            text_id = text_file.stem
            input_text = self.read_text_from_file(text_file)

            summary_openai = self.generate_summary("openai", input_text)
            if summary_openai:
                self.write_text_to_file(summary_openai, self.summaries_dir_openai / f"{text_id}.txt")

            summary_langchain = self.generate_summary("langchain-openai", input_text)
            if summary_langchain:
                self.write_text_to_file(summary_langchain, self.summaries_dir_langchain / f"{text_id}.txt")

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)

    processor = SummaryProcessor(config)
    processor.process_texts()
