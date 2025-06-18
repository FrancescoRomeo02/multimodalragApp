from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import io

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Inizializza il modello CLIP per embedding di immagini
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando device: {self.device}")
        
        # Carica modello e processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        print(f"Modello {model_name} caricato con successo")

# Inizializza il modello globale
embedding_model = ImageEmbedder()

# Funzione di utilità per convertire PIL Image in bytes (mantenuta per compatibilità)
def pil_image_to_bytes(image, format='PNG'):
    """
    Converte un oggetto PIL Image in bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def embed_image_direct(image_path):
    """
    Embedding diretto di un'immagine PNG
    """
    # Carica l'immagine
    image = Image.open(image_path)
    
    # Converti in RGB se necessario (PNG potrebbe avere canale alpha)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocessa l'immagine per CLIP
    inputs = embedding_model.processor(images=image, return_tensors="pt").to(embedding_model.device)
    
    # Genera l'embedding
    with torch.no_grad():
        image_features = embedding_model.model.get_image_features(**inputs)
        # Normalizza l'embedding per confronti più accurati
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Converti in numpy array
    embedding = image_features.cpu().numpy().flatten()
    
    return embedding

def embed_image_with_document(image_path, text_description=None):
    """
    Embedding usando metadati aggiuntivi
    """
    # Carica l'immagine
    image = Image.open(image_path)
    
    # Converti in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Crea metadati
    metadata = {
        'file_path': image_path,
        'image_size': image.size,
        'image_mode': image.mode,
        'text_description': text_description
    }
    
    # Genera embedding dell'immagine
    inputs = embedding_model.processor(images=image, return_tensors="pt").to(embedding_model.device)
    
    with torch.no_grad():
        image_features = embedding_model.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    embedding = image_features.cpu().numpy().flatten()
    
    # Se c'è una descrizione testuale, genera anche l'embedding del testo
    text_embedding = None
    if text_description:
        text_inputs = embedding_model.processor(text=text_description, return_tensors="pt").to(embedding_model.device)
        with torch.no_grad():
            text_features = embedding_model.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features.cpu().numpy().flatten()
    
    return embedding, metadata, text_embedding

def embed_multiple_images(image_paths):
    """
    Embedding di multiple immagini in batch
    """
    embeddings = []
    
    for image_path in image_paths:
        try:
            print(f"Processando: {image_path}")
            
            # Genera embedding
            embedding = embed_image_direct(image_path)
            
            embeddings.append({
                'path': image_path,
                'embedding': embedding,
                'shape': embedding.shape
            })
            
            print(f"✓ Completato: {image_path}")
            
        except Exception as e:
            print(f"✗ Errore nel processare {image_path}: {e}")
            continue
    
    return embeddings

def embed_text(text):
    """
    Embedding di testo per confronti multimodali
    """
    inputs = embedding_model.processor(text=text, return_tensors="pt").to(embedding_model.device)
    
    with torch.no_grad():
        text_features = embedding_model.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().flatten()

def save_embeddings(embeddings, output_file):
    """
    Salva gli embeddings in un file numpy
    """
    np.save(output_file, embeddings)
    print(f"Embeddings salvati in: {output_file}")

def calculate_image_similarity(embedding1, embedding2):
    """
    Calcola la similarità coseno tra due embeddings
    """
    # Normalizza gli embeddings se non già normalizzati
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Calcola similarità coseno (gli embeddings sono già normalizzati)
    similarity = np.dot(emb1, emb2)
    
    return similarity

def compare_images(image_path1, image_path2):
    """
    Confronto tra due immagini usando gli embeddings
    """
    print(f"Confrontando {image_path1} e {image_path2}...")
    
    embedding1 = embed_image_direct(image_path1)
    embedding2 = embed_image_direct(image_path2)
    
    similarity = calculate_image_similarity(embedding1, embedding2)
    
    print(f"Similarità tra {image_path1} e {image_path2}: {similarity:.4f}")
    
    return similarity

def compare_image_with_text(image_path, text_description):
    """
    Confronta un'immagine con una descrizione testuale
    """
    print(f"Confrontando immagine '{image_path}' con testo '{text_description}'...")
    
    image_embedding = embed_image_direct(image_path)
    text_embedding = embed_text(text_description)
    
    similarity = calculate_image_similarity(image_embedding, text_embedding)
    
    print(f"Similarità immagine-testo: {similarity:.4f}")
    
    return similarity

def analyze_image_content(image_path, possible_descriptions):
    """
    Analizza il contenuto di un'immagine confrontandola con possibili descrizioni
    """
    print(f"Analizzando contenuto di: {image_path}")
    
    image_embedding = embed_image_direct(image_path)
    results = []
    
    for description in possible_descriptions:
        text_embedding = embed_text(description)
        similarity = calculate_image_similarity(image_embedding, text_embedding)
        results.append({
            'description': description,
            'similarity': similarity
        })
    
    # Ordina per similarità decrescente
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("Risultati analisi:")
    for result in results:
        print(f"  '{result['description']}': {result['similarity']:.4f}")
    
    return results

# Esempio d'uso
if __name__ == "__main__":
    # Percorso dell'immagine PNG
    image_path = "/Users/fraromeo/Desktop/2.png"
    
    try:
        print("=== EMBEDDING IMMAGINE ===")
        # Metodo 1: Embedding diretto
        print("Generando embedding diretto...")
        embedding = embed_image_direct(image_path)
        print(f"✓ Embedding generato con shape: {embedding.shape}")
        print(f"✓ Range valori: [{embedding.min():.4f}, {embedding.max():.4f}]")
        
        print("\n=== EMBEDDING CON METADATI ===")
        # Metodo 2: Con descrizione
        embedding_meta, metadata, text_emb = embed_image_with_document(
            image_path, 
            text_description="Una immagine interessante"
        )
        print(f"✓ Embedding immagine: {embedding_meta.shape}")
        print(f"✓ Metadati: {metadata}")
        if text_emb is not None:
            print(f"✓ Embedding testo: {text_emb.shape}")
        
        print("\n=== CONFRONTO IMMAGINE-TESTO ===")
        # Esempio di confronto multimodale
        descriptions = [
            "a photo of a cat",
            "a picture of a dog", 
            "a landscape image",
            "a person smiling",
            "food on a table"
        ]
        
        analyze_image_content(image_path, descriptions)
        
        print(f"\n✓ Tutto completato con successo!")
            
    except FileNotFoundError:
        print(f"✗ File immagine non trovato: {image_path}")
    except Exception as e:
        print(f"✗ Errore generale: {e}")
        import traceback
        traceback.print_exc()

# INSTALLAZIONE RICHIESTA:
"""
pip install transformers torch torchvision pillow numpy
"""
