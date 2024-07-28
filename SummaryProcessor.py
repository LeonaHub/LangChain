import os
import json
import logging
import spacy
from pathlib import Path
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

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

    def split_text(self, text, max_length=1500):
        doc = nlp(text)
        segments = []
        current_segment = ""

        for sent in doc.sents:
            if len(current_segment) + len(sent.text) <= max_length:
                current_segment += sent.text + " "
            else:
                segments.append(current_segment.strip())
                current_segment = sent.text + " "
    
        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def update_context(self, old_context, new_segment, max_length=2000):
        combined_text = old_context + " " + new_segment
        doc = nlp(combined_text)
        updated_context = ""
        for sent in reversed(list(doc.sents)):
            if len(updated_context) + len(sent.text) > max_length:
                break
            updated_context = sent.text + " " + updated_context
        return updated_context.strip()

    def generate_summary(self, model_type, text, prev_context="", target_length=500):
        try:
            # Estimate token count from input text, adjust dynamically based on target length
            length_control = {"max_tokens": target_length}
            messages = [{"role": "system", "content": "Summarize this text:"}]
            if prev_context:
                messages.append({"role": "system", "content": "Previous context: " + prev_context})
            messages.append({"role": "user", "content": text})

            if model_type == "openai":
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    **{**self.summary_options, **length_control}
                )
                return response.choices[0].message.content
            elif model_type == "langchain-openai":
                model = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")
                response = model.invoke(messages, **{**self.summary_options, **length_control})
                return response.content
        except Exception as e:
            logger.error(f"Error generating summary with {model_type}: {e}")
            return None
        
    def call_model(self, model_type, messages, length_control):
        if model_type == "openai":
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                **{**self.summary_options, **length_control}
            )
            return response.choices[0].message.content
        elif model_type == "langchain-openai":
            model = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")
            response = model.invoke(messages, **{**self.summary_options, **length_control})
            return response.content
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
            
    def generate_summary(self, model_type, text, prev_context="", target_length=500):
        try:
            length_control = {"max_tokens": target_length}
            messages = [{"role": "system", "content": "Summarize this text:"}]
            if prev_context:
                messages.append({"role": "system", "content": "Previous context: " + prev_context})
            messages.append({"role": "user", "content": text})

            # Call the abstracted model invocation function
            return self.call_model(model_type, messages, length_control)
        except Exception as e:
            logger.error(f"Error generating summary with {model_type}: {e}")
            return None

    def process_texts(self):
        for text_file in self.data_dir.iterdir():
            text_id = text_file.stem
            input_text = self.read_text_from_file(text_file)
            ref_abstract_path = self.references_dir / f"{text_id}.txt"
            ref_abstract = self.read_text_from_file(ref_abstract_path)
            target_length = len(ref_abstract.split())  # Estimate target length based on reference abstract word count

            accumulated_context = ""
            segments = self.split_text(input_text)

            for segment in segments:
                summary_segment = self.generate_summary("langchain-openai", segment, accumulated_context, target_length)
                if summary_segment:
                    self.write_text_to_file(summary_segment, self.summaries_dir_langchain / f"{text_id}.txt")
                accumulated_context = self.update_context(accumulated_context, segment)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)

    processor = SummaryProcessor(config)
    processor.process_texts()
