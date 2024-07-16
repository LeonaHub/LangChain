import os
import requests
from pdfminer.high_level import extract_text

def download_pdf(url, filename):
    """
    Download PDF from a given URL and save it locally.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print("Failed to download the file.")
        return False

def extract_text_from_pdf(pdf_path, text_path):
    """
    Extract text from a given PDF file and save it to a text file.
    """
    try:
        text = extract_text(pdf_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text extracted and saved to {text_path}")
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")

def main():
    pdf_url = ''
    output_dir = 'raw_texts'
    pdf_filename = 'downloaded_article.pdf'
    text_filename = 'extracted_text.txt'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_path = os.path.join(output_dir, pdf_filename)
    text_path = os.path.join(output_dir, text_filename)
    
    if download_pdf(pdf_url, pdf_path):
        extract_text_from_pdf(pdf_path, text_path)

if __name__ == "__main__":
    main()
