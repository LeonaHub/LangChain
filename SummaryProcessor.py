import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from langchain import LangChain
from langchain.schema import SystemMessage, HumanMessage
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class SummaryProcessor:
    def __init__(self, config):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.data_dir = Path(config['data_dir'])
        self.summaries_dir_openai1 = Path(config['summaries_dir_openai1'])
        self.summaries_dir_langchain1 = Path(config['summaries_dir_langchain1'])
        self.summaries_dir_openai2 = Path(config['summaries_dir_openai12'])
        self.summaries_dir_langchain2 = Path(config['summaries_dir_langchain12'])
        self.references_dir = Path(config['references_dir'])
        self.summary_options = config['summary_options']
        self.openai_client = OpenAI(api_key=self.api_key)
        self.langchain = LangChain(api_key=self.api_key, model="gpt-3.5-turbo", stateful=True)
        self.setup_directories()

    def setup_directories(self):
        self.summaries_dir_openai1.mkdir(parents=True, exist_ok=True)
        self.summaries_dir_langchain1.mkdir(parents=True, exist_ok=True)
        self.summaries_dir_openai2.mkdir(parents=True, exist_ok=True)
        self.summaries_dir_langchain2.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

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

    def split_text(self, text, max_length=5000):
        doc = nlp(text)
        segments = []
        current_segment = ""
        current_length = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if current_length + len(sent_text) > max_length:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sent_text
                current_length = len(sent_text)
            else:
                current_segment += " " + sent_text if current_segment else sent_text
                current_length += len(sent_text)

        if current_segment:
            segments.append(current_segment)

        return segments

    def process_text_with_models(self, segment):
        # Process with LangChain-enhanced GPT-3.5
        langchain_response = self.langchain.process_text(segment, additional_messages=[
            SystemMessage(content="Summarize this text:"),
            HumanMessage(content=segment)
        ], **self.summary_options)

        # Process with standalone GPT-3.5
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this text:"},
                {"role": "user", "content": segment}
            ],
            **self.summary_options
        )

        return langchain_response, response.choices[0].message.content if response else None

    def process_texts(self):
        for text_file in self.data_dir.iterdir():
            text_id = text_file.stem
            input_text = self.read_text_from_file(text_file)
            segments = self.split_text(input_text)

            first_round_summaries_langchain = []
            first_round_summaries_openai = []

            self.langchain.reset()  # Resetting only at the beginning of a new document
            for segment in segments:
                langchain_summary, openai_summary = self.process_text_with_models(segment)
                first_round_summaries_langchain.append(langchain_summary if langchain_summary else "")
                first_round_summaries_openai.append(openai_summary if openai_summary else "")

            # Combine all segment summaries for the second round for Langchain
            combined_summary_langchain = ' '.join(first_round_summaries_langchain)
            if combined_summary_langchain:
                self.write_text_to_file(combined_summary_langchain, self.summaries_dir_langchain1 / f"{text_id}.txt")

            second_round_langchain_summary = self.langchain.process_text(combined_summary_langchain, additional_messages=[
                SystemMessage(content="Summarize this combined text:")
            ])
            if second_round_langchain_summary:
                self.write_text_to_file(second_round_langchain_summary, self.summaries_dir_langchain2 / f"{text_id}.txt")

            # Combine all segment summaries for the second round for OpenAI
            combined_summary_openai = ' '.join(first_round_summaries_openai)
            if combined_summary_openai:
                self.write_text_to_file(combined_summary_openai, self.summaries_dir_openai1 / f"{text_id}.txt")

            second_round_openai_summary = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Summarize this combined text:"},
                          {"role": "user", "content": combined_summary_openai}],
                **self.summary_options
            )
            if second_round_openai_summary:
                self.write_text_to_file(second_round_openai_summary.choices[0].message.content, self.summaries_dir_openai2 / f"{text_id}.txt")

            # Reset after processing the document
            self.langchain.reset()

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)

    processor = SummaryProcessor(config)
    processor.process_texts()
