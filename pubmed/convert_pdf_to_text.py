import os
import logging
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text_to_fp
import io

def convert_pdf_to_text(pdf_dir, text_dir):
    if not os.path.isdir(pdf_dir):
        logging.error(f"The directory {pdf_dir} does not exist.")
        return
    
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
        logging.info(f"Created directory {text_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logging.warning("No PDF files found in the directory.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        text_path = os.path.join(text_dir, filename.replace('.pdf', '.txt'))
        
        # 使用自定义的LAParams
        laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1, boxes_flow=0.5)
        
        try:
            output_string = io.StringIO()
            with open(pdf_path, 'rb') as in_file:
                extract_text_to_fp(in_file, output_string, laparams=laparams, output_type='text', codec='utf-8')
            text = output_string.getvalue()
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            logging.info(f"Converted {filename} to text at {text_path}.")
        except Exception as e:
            logging.error(f"Failed to convert {filename}: {e}")

pdf_dir = 'pdf'  
text_dir = 'pdf_text'  
convert_pdf_to_text(pdf_dir, text_dir)