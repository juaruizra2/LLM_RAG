import pandas as pd
import warnings
import os
import pickle
from embeddings_factory import EmbeddingFactory
from preprocessing import TextPreprocessor
from scipy.spatial.distance import cosine
from config import embedding_to_use, embedding_models, similarity_threshold, prompt
from openai import OpenAI

# Deshabilitar warnings específicos
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Configurar tokenizers para evitar warnings de paralelismo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAG:
    def __init__(self, knowledge_base, preprocessor, embedding_type='flag', model_name=None):
        self.knowledge_base = knowledge_base
        self.preprocessor = preprocessor
        self.embedding_type = embedding_type
        self.model_name = model_name
        
        self.embedding_generator = EmbeddingFactory.create_generator(
            self.embedding_type, 
            model_name=self.model_name, 
            use_fp16=True
        )
    
    def get_similarity(self, query_embedding, knowledge_base_embedding):
        self.knowledge_base["distances"] = knowledge_base_embedding.apply(lambda x: cosine(query_embedding, x))
        return self.knowledge_base

    def transform_distances(self, df_embedding_copy):
        """Transforma las distancias aplicando 1 - distances para convertir a similitud."""
        df_embedding_copy["distances"] = 1 - df_embedding_copy["distances"]
        return df_embedding_copy

    def filter_by_similarity_threshold(self, df_embedding_copy, threshold=0.70):
        """Filtra por similitud mayor o igual al umbral especificado."""
        return df_embedding_copy[df_embedding_copy["distances"] >= threshold].reset_index(drop=True).head(2)

    def get_context(self, df_embedding_copy):
        """Concatena todos los plots de los casos filtrados en un solo string."""
        if df_embedding_copy.empty:
            return ""
        
        context = df_embedding_copy["plot"].tolist()
        urls = df_embedding_copy["image"].tolist()

        end_context = [
            {'contexto': contexto, 'image_url': url} 
            for contexto, url in zip(context, urls)
        ]

        return end_context

    def generate_response(self, prompt_formatted):
        """Genera una respuesta usando OpenAI basada en el prompt formateado."""
        try:
            import os
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            client = OpenAI()
            
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_formatted
                    }
                ]
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"

    def filter_by_threshold(self, threshold=0.90):
        return self.knowledge_base[self.knowledge_base["distances"] >= threshold]

    def query(self, query):
        query = self.preprocessor.preprocess(query)
        query_embedding = self.embedding_generator.generate_embedding(query)
        df_embedding_copy = self.knowledge_base.copy()
        df_embedding_copy = self.get_similarity(query_embedding, df_embedding_copy["embedding"])
        df_embedding_copy = df_embedding_copy.sort_values(by="distances", ascending=True).reset_index(drop=True)
        df_embedding_copy = self.transform_distances(df_embedding_copy)
        df_embedding_copy = self.filter_by_similarity_threshold(df_embedding_copy, similarity_threshold)
        context = self.get_context(df_embedding_copy)
        prompt_formatted = prompt.format(context=context, question=query)
        response = self.generate_response(prompt_formatted)
        return response, context

if __name__ == "__main__":
    
    knowledge_base = pd.read_parquet(f"data/movies-dataset-embeddings-{embedding_to_use}.parquet")
    preprocessor = TextPreprocessor()
    model_name = embedding_models[embedding_to_use]["model"]
    rag = RAG(knowledge_base, preprocessor, embedding_type=embedding_to_use, model_name=model_name)
    
    # Guardar la instancia RAG en un pickle
    print("Guardando instancia RAG en pickle...")
    with open('rag_model.pickle', 'wb') as f:
        pickle.dump(rag, f)
    print("Instancia RAG guardada exitosamente en 'rag_model.pickle'")

    """
    
    print("Welcome to the RAG system")

    questions = [
        "What is the name of the movie where humans and AIs coexist and have a battle for control of reality?",
        "In the movie Enigma, what is the name of the protagonist and who plays the CIA agent?",
        "What is the movie Echoes of Tomorrow about? Show the answer by indicating the context of the question. Example: This movie is about…",
        "Show the image related to the movie Stellar Odyssey"
    ]

    for question in questions:
        print("--------------------------------")
        print("Question: ", question)
        response, context = rag.query(question)
        print("--------------------------------")
        print("Response: ", response)
        print("--------------------------------")
    """