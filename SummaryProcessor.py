import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import GPT2Tokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
        self.summaries_dir_openai2 = Path(config['summaries_dir_openai2'])
        self.summaries_dir_langchain2 = Path(config['summaries_dir_langchain2'])
        self.references_dir = Path(config['references_dir'])
        self.summary_options = config['summary_options']
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.openai_client = OpenAI(api_key=self.api_key)
        langchain_model = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")
        self.parser = StrOutputParser()
        self.langchain = langchain_model | self.parser
        self.setup_directories()
        # Setup PromptTemplate
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", "Summarize the text above.")]
        )

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

    def get_token_count(self, text):
        return len(self.tokenizer.encode(text))

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


    def process_text_with_openai(self, segment):
        # Process with standalone GPT-3.5
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize this text:"},
                {"role": "user", "content": segment}
            ],
            **self.summary_options
        )

        return response.choices[0].message.content if response else None
    
    def process_texts(self):
        for text_file in self.data_dir.iterdir():
            text_id = text_file.stem
            input_text = self.read_text_from_file(text_file)
            reference_abstract = self.read_text_from_file(self.references_dir / f"{text_id}.txt")
            target_length = self.get_token_count(reference_abstract)
            self.summary_options['max_tokens'] = target_length

            segments = self.split_text(input_text)

            first_round_summaries_openai = []
             # Accumulate all segments as messages for langchain_openai model
            messages = []
           
            self.openai_client = OpenAI(api_key=self.api_key)
            self.langchain = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")

            for segment in segments:
                openai_summary = self.process_text_with_openai(segment)
                first_round_summaries_openai.append(openai_summary if openai_summary else "")

            # Combine all segment summaries for the second round for OpenAI
            combined_summary_openai = ' '.join(first_round_summaries_openai)
            second_round_openai_summary = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Summarize this combined text:"},
                          {"role": "user", "content": combined_summary_openai}],
                **self.summary_options
            )
            if second_round_openai_summary:
                self.write_text_to_file(second_round_openai_summary.choices[0].message.content, self.summaries_dir_openai2 / f"{text_id}.txt")
                
            # Accumulate messages for LangChain model
            messages = [HumanMessage(content=segment) for segment in segments]
            prompt_result = self.prompt_template.invoke({})  # Using an empty dict since there are no variables to replace
            messages.extend(prompt_result.to_messages())

            # Process with LangChain
            langchain_summary = self.langchain.invoke(messages, **self.summary_options)
            if langchain_summary:
                self.write_text_to_file(langchain_summary.content, self.summaries_dir_langchain2 / f"{text_id}.txt")


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)

    processor = SummaryProcessor(config)
    processor.process_texts()
