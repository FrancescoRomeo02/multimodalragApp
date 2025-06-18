from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict, Union
from PIL import Image
from io import BytesIO
import base64
from app.config import EMBEDDING_MODEL_NAME

class QdrantCompatibleEmbedder:
    def __init__(self):
        self.model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            device="cpu",
            trust_remote_code=True
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.get_text_embedding(text) for text in texts]
    
    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(input, str):
            return self.model.get_text_embedding(input)
        return self.embed_documents(input)
    
    #Embedding della query
    def embed_query(self, text: str) -> List[float]:
        return self.model.get_text_embedding(text)
    #Embedding dell'immagine (to check)
    def embed_images(self, image_dicts: List[Dict]) -> List[Dict]:
        results = []
        for img_dict in image_dicts:
            try:
                img_data = base64.b64decode(img_dict["image_base64"])
                img = Image.open(BytesIO(img_data))
                embedding = self.model.get_image_embedding(img)
                img_dict["embedding"] = embedding
                results.append(img_dict)
            except Exception as e:
                print(f"Errore embedding immagine: {str(e)}")
        return results

# Usato nel retriever per l'embedding
def get_embedding_model() -> QdrantCompatibleEmbedder:
    return QdrantCompatibleEmbedder()