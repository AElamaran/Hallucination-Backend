from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from vector_db_populate import clear_vector_db, populate_db
from rag_service_query import response_rag_pipeline,response_foundation_model,analyze_semantic_entropy,pure_response_rag_pipeline
from flask_cors import CORS
import os
import json
from output import generate_final_prompt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
DATA_PATH = 'knowledge-base'
VECTOR_DB_PATH = 'database'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH)

app.config['DATA_PATH'] = DATA_PATH
app.config['VECTOR_DB_PATH'] = VECTOR_DB_PATH

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/clear-db', methods=['GET'])
def clear_db():
    try:
        clear_vector_db()
        return jsonify({'message': 'DB cleared'}), 200
    except:
        return jsonify({'message': 'Error clearing DB'}), 500

@app.route('/populate-db', methods=['POST'])
def populate_vector_db():
    try:
        message = populate_db()
        add_db_message = message[1]
        return jsonify({'success': 'File successfully uploaded', 'db_message': add_db_message}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-db-job', methods=['GET'])
def clear_db_job():
    try:
        clear_vector_db()
        return jsonify({'message': 'DB cleared'}), 200
    except:
        return jsonify({'message': 'Error clearing DB'}), 500

@app.route('/rag_response', methods=['POST'])
def handle_rag_query():
    try:
        request_data = request.get_json()
        query_text = request_data.get('query_text', '')
        if query_text:
            response_text = response_rag_pipeline(query_text)
            response_type=response_text.get('response_type','')
            if (response_type=="The Model's Response is Unlikely to be Hallucinate therefore Approaching RAG Approach"):
                return jsonify({'response': response_text})
            else:
                final_answer = generate_final_prompt(response_text)
                return jsonify({'response': final_answer}), 200
        else:
            return jsonify({'error': 'No query text provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/foundation_response', methods=['POST'])
def handle_raw_query():
    try:
        request_data = request.get_json()
        query_text = request_data.get('query_text', '')
        if query_text:
            response_text = response_foundation_model(query_text)
            return jsonify({'response': response_text}), 200
        else:
            return jsonify({'error': 'No query text provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/semantic_entropy_analysis', methods=['POST'])
def handle_semantic_entropy_analysis():
    try:
        request_data = request.get_json()
        query_text = request_data.get('query_text', '')
        if query_text:
            analysis_report = analyze_semantic_entropy(query_text)
            return jsonify({'analysis_report': analysis_report}), 200
        else:
            return jsonify({'error': 'No query text provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/pure_rag_response', methods=['POST'])
def handle_pure_rag_query():
    try:
        request_data = request.get_json()
        query_text = request_data.get('query_text', '')
        if query_text:
            response_text = pure_response_rag_pipeline(query_text)
            return jsonify({'response': response_text}), 200
        else:
            return jsonify({'error': 'No query text provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, timeout=10000)
