from flask import Flask, jsonify, send_from_directory
from pymongo import MongoClient
import os

app = Flask(__name__)
from flask_cors import CORS
CORS(app)  # This allows the frontend to talk to this backend without security errors

# 1. CONNECT TO DB
client = MongoClient('mongodb://localhost:27017/')
db = client['traffic_violations_db']
collection = db['violations']

# IMPORTANT: Point this to the folder where your 'output' folder lives
# Example: If images are in C:/Users/Ahmed/TrafficProject/output/...
# You should point to C:/Users/Ahmed/TrafficProject
BASE_IMAGE_FOLDER = os.path.abspath("../") 

@app.route('/violations', methods=['GET'])
def get_violations():
    # Fetch data
    data = list(collection.find({}, {'_id': 0}))
    
    # Update image paths to be "server links" instead of "hard drive paths"
    # The frontend will receive: http://localhost:5000/images/output/cropped_photos/...
    for record in data:
        original_path = record.get('Plate_Image_Path', '')
        # We strip the leading '../' to make it a clean URL segment
        clean_path = original_path.replace('../', '').replace('\\', '/')
        record['Plate_Image_Path'] = f"http://localhost:5000/images/{clean_path}"
        
    return jsonify(data)

# This route serves the actual image files
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(BASE_IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)