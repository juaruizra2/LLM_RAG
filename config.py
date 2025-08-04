embedding_to_use = "sentence_transformer"

embedding_models = {
        "flag": {
            "model": "BAAI/bge-m3"
        },
        "sentence_transformer": {
            "model": "BAAI/bge-small-en-v1.5"
        },
        "openai": {
            "model": "text-embedding-3-small"
        }
    }


similarity_threshold = 0.55

prompt = """

You are an expert assistant in answering questions based on a given question context.

QUESTION:
{question}

QUESTION CONTEXT:
{context}

Here are the instructions you must follow:

- Use the provided question context to answer the question.
- If the context is insufficient or unrelated to the question, respond that you do not have information on the topic.
- If the question requests an image, reply with the image URL provided in image_url.

"""
    
