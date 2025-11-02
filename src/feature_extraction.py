import os
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.config import Config

class FeatureExtractor:
    def __init__(self):
        print(f"Loading Sentence-BERT model: {Config.MODEL_NAME}")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        
        # Move model to GPU
        self.model = self.model.to(Config.DEVICE)
        print(f"Model loaded on {Config.DEVICE}")
    
    def encode_texts(self, texts, batch_size=None):
        """Generate embeddings for texts using GPU"""
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        
        print(f"Encoding {len(texts)} texts on GPU...")
        
        # Convert to list if single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings on GPU
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=Config.DEVICE,
                show_progress_bar=True
            )
        
        # Convert to numpy array
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"Encoding complete. Shape: {embeddings_np.shape}")
        return embeddings_np
    
    def calculate_cosine_similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity between embeddings"""
        # Convert to tensors if needed
        if not isinstance(embeddings1, torch.Tensor):
            embeddings1 = torch.tensor(embeddings1).to(Config.DEVICE)
        if not isinstance(embeddings2, torch.Tensor):
            embeddings2 = torch.tensor(embeddings2).to(Config.DEVICE)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings1, embeddings2)
        
        return similarity.cpu().numpy()
    
    def save_embeddings(self, embeddings, filename):
        """Save embeddings to file"""
        import os
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        filepath = os.path.join(Config.MODEL_DIR, filename)
        np.save(filepath, embeddings)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filename):
        """Load embeddings from file"""
        filepath = os.path.join(Config.MODEL_DIR, filename)
        if os.path.exists(filepath):
            embeddings = np.load(filepath)
            print(f"Embeddings loaded from {filepath}")
            return embeddings
        return None
