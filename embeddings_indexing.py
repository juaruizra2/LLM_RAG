"""This script is used to create a knowledge base from a csv file and save it as a parquet file.
It also creates embeddings for the text in the plot column and saves them as a column in the parquet file.
The text is preprocessed before generating embeddings.
"""


import os
import pandas as pd
from preprocessing import TextPreprocessor
from tqdm import tqdm
from embeddings_factory import EmbeddingFactory
from config import embedding_to_use, embedding_models

if __name__ == "__main__":
    parquet_path = "data/movies-dataset.parquet"
    csv_path = "data/movies-dataset.csv"
    
    if os.path.exists(parquet_path):
        print("Using existing parquet file...")
        knowledge_base = pd.read_parquet(parquet_path)
    else:
        print("Parquet file not found. Reading CSV and creating Parquet...")
        knowledge_base = pd.read_csv(csv_path)
        knowledge_base.to_parquet(parquet_path)
        print(f"Parquet file saved in: {parquet_path}")
    
    
    print("Total rows in knowledge base: ", len(knowledge_base))
    print("Columns in knowledge base: ", knowledge_base.columns)

    
    knowledge_base["plot"] = knowledge_base["title"] + " . " + knowledge_base["plot"]
    
    preprocessor = TextPreprocessor()
    tqdm.pandas(desc="Preprocessing text...")
    knowledge_base["plot_clean"] = knowledge_base["plot"].progress_apply(preprocessor.preprocess)
    
    tqdm.pandas(desc="Generating embeddings...")
    model = embedding_models[embedding_to_use]["model"]
    
    embedding_generator = EmbeddingFactory.create_generator(embedding_to_use, model_name=model, use_fp16=True)
    knowledge_base["embedding"] = knowledge_base["plot_clean"].progress_apply(embedding_generator.generate_embedding)
    
    print("Saving to parquet file...")
    knowledge_base.to_parquet(f"data/movies-dataset-embeddings-{embedding_to_use}.parquet")

    print("Embeddings generated and saved to parquet file.")

