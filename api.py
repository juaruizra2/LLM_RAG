import pickle
import pandas as pd
import warnings
import os
from rag import RAG
from flask import Flask, request, jsonify

#puerto 5000
rag = pickle.load(open('rag_model.pickle', 'rb'))

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/query', methods=['GET'])
def query():
    # Forma simple con manejo de error
    try:
        data = request.get_json()
        question = data.get('question')
        print(question)
        response, _ = rag.query(question)
        return jsonify({"response": response})
    except KeyError:
        return jsonify({"error": "Falta el parámetro 'question'"}), 400

if __name__ == '__main__':
    app.run(debug=True)
