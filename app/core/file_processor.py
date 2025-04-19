from PyPDF2 import PdfReader
import docx
import fitz  
from PIL import Image
import pytesseract
import io

from ..config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FileProcessor:
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    @staticmethod
    def get_pdf_text(pdf_path):
        try:
            
            pdf_reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            if text.strip(): 
                return text
            else:
                
                return FileProcessor.ocr_pdf_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"[ERROR] PDF reading failed: {e}")
            return FileProcessor.ocr_pdf_with_pymupdf(pdf_path)

    @staticmethod
    def ocr_pdf_with_pymupdf(pdf_path):
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img) + "\n"
        return text

    @staticmethod
    def get_docx_text(docx_path):
        doc = docx.Document(docx_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)

    @staticmethod
    def get_txt_text(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"[ERROR] TXT file reading failed: {e}")
            return ""

    @staticmethod
    def get_file_text(file_path):
        if file_path.lower().endswith('.pdf'):
            return FileProcessor.get_pdf_text(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            return FileProcessor.get_docx_text(file_path)
        elif file_path.lower().endswith('.txt'):
            return FileProcessor.get_txt_text(file_path)
        return ""

    @staticmethod
    def get_text_chunks(text):
        return FileProcessor.TEXT_SPLITTER.split_text(text) if text else []