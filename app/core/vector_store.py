import os
from ..config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store_dir = Config.VECTOR_STORE_DIR
    
    def create_vector_store(self, text_chunks, db_name):
        if not text_chunks:
            return False

        pdf_vector_path = os.path.join(self.vector_store_dir, db_name)
        os.makedirs(pdf_vector_path, exist_ok=True)
        
        if os.path.exists(os.path.join(pdf_vector_path, "index.faiss")):
            vector_store = FAISS.load_local(pdf_vector_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_texts(text_chunks)
        else:
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        
        vector_store.save_local(pdf_vector_path)
        return True
    
    def get_vector_store(self, db_name):
        db_path = os.path.join(self.vector_store_dir, db_name)
        return FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True) \
               if os.path.exists(db_path) else None












# import os
# import json
# from pathlib import Path
# from typing import List, Dict, Optional
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import openai
# from ..config import Config

# class VectorStoreManager:
#     def __init__(self):
#         # Convert to absolute path and normalize
#         self.vector_store_dir = Path(Config.VECTOR_STORE_DIR).absolute()
#         print(f"Vector store directory set to: {self.vector_store_dir}")
        
#         # Create directory if it doesn't exist
#         self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
#         # Verify directory exists and is writable
#         if not os.access(self.vector_store_dir, os.W_OK):
#             raise PermissionError(f"Cannot write to directory: {self.vector_store_dir}")

#     def create_vector_store(self, text_chunks: List[str], db_name: str) -> bool:
#         try:
#             # Sanitize the db_name to prevent path traversal
#             safe_db_name = "".join(c for c in db_name if c.isalnum() or c in ('_', '-'))
#             db_path = self.vector_store_dir / f"{safe_db_name}.json"
            
#             print(f"Attempting to create vector store at: {db_path}")

#             # Generate embeddings
#             embeddings = []
#             for i, text in enumerate(text_chunks, 1):
#                 response = openai.Embedding.create(
#                     input=text,
#                     engine=Config.EMBEDDING_MODEL
#                 )
#                 embeddings.append({
#                     "text": text,
#                     "embedding": response['data'][0]['embedding']
#                 })
#                 if i % 5 == 0:
#                     print(f"Processed {i}/{len(text_chunks)} chunks")

#             # Save to file
#             with open(db_path, 'w') as f:
#                 json.dump(embeddings, f)
            
#             print(f"Successfully saved vector store with {len(embeddings)} entries")
#             return True

#         except Exception as e:
#             print(f"Error creating vector store: {str(e)}")
#             return False

#     def get_vector_store(self, db_name: str) -> Optional[List[Dict]]:
#         safe_db_name = "".join(c for c in db_name if c.isalnum() or c in ('_', '-'))
#         db_path = self.vector_store_dir / f"{safe_db_name}.json"
        
#         print(f"Looking for vector store at: {db_path}")
        
#         if not db_path.exists():
#             print("Vector store file does not exist")
#             return None
            
#         try:
#             with open(db_path, 'r') as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"Error loading vector store: {str(e)}")
#             return None