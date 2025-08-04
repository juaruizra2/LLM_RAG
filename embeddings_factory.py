

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from typing import Optional, Union, List
import numpy as np
import os


class EmbeddingGenerator(ABC):
    """
    Clase abstracta para generar embeddings de texto.
    Implementa el patrón Strategy para permitir diferentes métodos de embeddings.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Inicializa el generador de embeddings.
        
        Args:
            model_name: Nombre del modelo a usar
            **kwargs: Parámetros adicionales específicos del modelo
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model(**kwargs)
    
    @abstractmethod
    def _initialize_model(self, **kwargs):
        """Inicializa el modelo específico de embeddings."""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera embeddings para un texto dado.
        
        Args:
            text: Texto para generar embeddings
            
        Returns:
            Array numpy con los embeddings
        """
        pass
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos para generar embeddings
            
        Returns:
            Lista de arrays numpy con los embeddings
        """
        return [self.generate_embedding(text) for text in texts]


class FlagEmbeddingGenerator(EmbeddingGenerator):
    """
    Generador de embeddings usando FlagEmbedding (BGE-M3).
    """
    
    def _initialize_model(self, **kwargs):
        """Inicializa el modelo BGEM3FlagModel."""
        try:
            use_fp16 = kwargs.get('use_fp16', True)
            self.model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
        except Exception as e:
            raise RuntimeError(f"Error inicializando FlagEmbedding model: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera embeddings usando FlagEmbedding.
        
        Args:
            text: Texto para generar embeddings
            
        Returns:
            Array numpy con los embeddings
        """
        try:
            if self.model is None:
                raise RuntimeError("Modelo no inicializado")
            
            embeddings = self.model.encode([text])['dense_vecs']
            return embeddings[0]  # Retorna el primer (y único) embedding como array 1D
        except Exception as e:
            raise RuntimeError(f"Error generando embeddings con FlagEmbedding: {e}")


class SentenceTransformerGenerator(EmbeddingGenerator):
    """
    Generador de embeddings usando SentenceTransformer.
    """
    
    def _initialize_model(self, **kwargs):
        """Inicializa el modelo SentenceTransformer."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Error inicializando SentenceTransformer model: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera embeddings usando SentenceTransformer.
        
        Args:
            text: Texto para generar embeddings
            
        Returns:
            Array numpy con los embeddings
        """
        try:
            if self.model is None:
                raise RuntimeError("Modelo no inicializado")
            
            embeddings = self.model.encode([text])[0]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generando embeddings con SentenceTransformer: {e}")


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """
    Generador de embeddings usando OpenAI API.
    """
    
    def _initialize_model(self, **kwargs):
        """Inicializa el cliente de OpenAI."""
        import os
        os.environ["OPENAI_API_KEY"] = "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"
        self.client = OpenAI()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Genera embeddings usando OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embeddings = np.array(response.data[0].embedding)
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generando embeddings con OpenAI: {e}")


class EmbeddingFactory:
    """
    Factory para crear diferentes tipos de generadores de embeddings.
    """
    
    @staticmethod
    def create_generator(embedding_type: str, model_name: Optional[str] = None, **kwargs) -> EmbeddingGenerator:
        """
        Crea un generador de embeddings según el tipo especificado.
        
        Args:
            embedding_type: Tipo de embeddings ('flag', 'sentence_transformer', 'openai', etc.)
            model_name: Nombre del modelo (opcional, usa defaults si no se especifica)
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            Instancia de EmbeddingGenerator
        """
        if embedding_type.lower() == 'flag':
            model_name = model_name or 'BAAI/bge-m3'
            return FlagEmbeddingGenerator(model_name, **kwargs)
        
        elif embedding_type.lower() == 'sentence_transformer':
            model_name = model_name or 'BAAI/bge-small-en-v1.5'
            return SentenceTransformerGenerator(model_name, **kwargs)
        
        elif embedding_type.lower() == 'openai':
            model_name = model_name or 'text-embedding-3-small'
            return OpenAIEmbeddingGenerator(model_name, **kwargs)
        
        else:
            raise ValueError(f"Tipo de embeddings no soportado: {embedding_type}")