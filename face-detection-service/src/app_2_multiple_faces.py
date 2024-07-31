from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from flask_cors import CORS
import logging
from deepface import DeepFace
from config import (ALLOWED_EXTENSIONS, FOLDER_NAME, HAARCASCADE_FRONTALFACE_XML,
                    THRESHOLD_FOR_IS_IMAGE_CLEAR, THRESHOLD_FOR_COMPARE_FACES,
                    MODEL_NAME_FOR_COMPARE_FACES, DISTANCE_METRIC_FOR_COMPARE_FACES,
                    EXTRACTED_IMAGES_FOLDER, PDF_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
                    FACE1_IMAGE, FACE2_IMAGE, EXTRACTED_VIDEO_IMAGE)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', FOLDER_NAME))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_extracted_images(images, output_folder):
    image_paths = []
    for image_index, img in enumerate(images):
        image_ext = 'png'
        image_path = os.path.join(output_folder, f"extracted_image_{image_index}.{image_ext}")
        cv2.imwrite(image_path, np.array(img))
        image_paths.append(image_path)
    return image_paths

def extract_image_from_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    if success:
        cv2.imwrite(output_path, image)
        video.release()
        return image
    video.release()
    return None

def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAARCASCADE_FRONTALFACE_XML)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return face_images

def is_image_clear(image_path, threshold=THRESHOLD_FOR_IS_IMAGE_CLEAR):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    resized_image = cv2.resize(normalized_image, (224, 224))
    equalized = cv2.equalizeHist(resized_image)
    denoised = cv2.fastNlMeansDenoising(equalized, h=10)
    return denoised

def compare_faces(face1_path, face2_paths, threshold=THRESHOLD_FOR_COMPARE_FACES, model_name=MODEL_NAME_FOR_COMPARE_FACES, distance_metric=DISTANCE_METRIC_FOR_COMPARE_FACES):
    try:
        best_match = {'distance': float('inf'), 'accuracy': 0, 'faces_are_same': False}
        for face2_path in face2_paths:
            result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, model_name=model_name, distance_metric=distance_metric)
            score = result['distance']
            is_match = score < threshold

            if score < best_match['distance']:
                best_match = {
                    'distance': score,
                    'accuracy': 1 - score,
                    'faces_are_same': is_match
                }

        return best_match
    except Exception as e:
        logging.error(f"Error comparing faces: {e}")
        return {'error': 'Failed to compare faces'}

@app.route('/process-media', methods=['POST'])
def process_media():
    if 'pdf' not in request.files or 'video' not in request.files:
        return jsonify({'error': 'Two files are required'}), 400
    
    file1 = request.files['pdf']
    file2 = request.files['video']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Files cannot be empty'}), 400
    
    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    
    file1_filename = secure_filename(file1.filename)
    file2_filename = secure_filename(file2.filename)
    
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1_filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2_filename)
    
    file1.save(file1_path)
    file2.save(file2_path)
    
    if not os.path.exists(file1_path) or os.path.getsize(file1_path) == 0:
        return jsonify({'error': 'Files cannot be empty'}), 400
    
    if not os.path.exists(file2_path) or os.path.getsize(file2_path) == 0:
        return jsonify({'error': 'Files cannot be empty'}), 400
    
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], EXTRACTED_IMAGES_FOLDER)
    os.makedirs(output_folder, exist_ok=True)
    
    if file1.filename.lower().endswith(PDF_EXTENSIONS) and file2.filename.lower().endswith(VIDEO_EXTENSIONS):
        images = convert_from_path(file1_path)
        if not images:
            return jsonify({'error': 'No images found in the PDF file/Upload correct PDF file'}), 400
        
        image_paths = save_extracted_images(images, output_folder)
        
        clear_image_paths = [path for path in image_paths if is_image_clear(path)]
        if not clear_image_paths:
            return jsonify({'error': 'No clear images found in the PDF file. Please upload a clearer document.'}), 400

        video_image_path = os.path.join(output_folder, EXTRACTED_VIDEO_IMAGE)
        video_image = extract_image_from_video(file2_path, video_image_path)
        
        if video_image is None:
            return jsonify({'error': 'Failed to extract image from video'}), 500

        faces_from_pdf = []
        for image_path in clear_image_paths:
            faces_from_pdf.extend(detect_faces(image_path))
        
        if not faces_from_pdf:
            return jsonify({'error': 'No faces found in the PDF file'}), 400
        
        faces_from_video = detect_faces(video_image_path)
        
        if not faces_from_video:
            return jsonify({'error': 'No faces found in the video'}), 400
        
        face1_path = os.path.join(output_folder, FACE1_IMAGE)
        cv2.imwrite(face1_path, preprocess_image(faces_from_pdf[0]))
        
        face2_paths = []
        for face in faces_from_video:
            face2_path = os.path.join(output_folder, FACE2_IMAGE)
            cv2.imwrite(face2_path, preprocess_image(face))
            face2_paths.append(face2_path)
        
        result = compare_faces(face1_path, face2_paths)
        return jsonify(result), 200
    
    elif file1.filename.lower().endswith(IMAGE_EXTENSIONS) and file2.filename.lower().endswith(VIDEO_EXTENSIONS):
        image_path = file1_path
        video_image_path = os.path.join(output_folder, EXTRACTED_VIDEO_IMAGE)
        video_image = extract_image_from_video(file2_path, video_image_path)
        
        if video_image is None:
            return jsonify({'error': 'Failed to extract image from video'}), 500

        faces_from_image = detect_faces(image_path)
        
        if not faces_from_image:
            return jsonify({'error': 'No faces found in the uploaded image'}), 400
        
        faces_from_video = detect_faces(video_image_path)
        
        if not faces_from_video:
            return jsonify({'error': 'No faces found in the video'}), 400
        
        face1_path = os.path.join(output_folder, FACE1_IMAGE)
        cv2.imwrite(face1_path, preprocess_image(faces_from_image[0]))
        
        face2_paths = []
        for face in faces_from_video:
            face2_path = os.path.join(output_folder, FACE2_IMAGE)
            cv2.imwrite(face2_path, preprocess_image(face))
            face2_paths.append(face2_path)
        
        result = compare_faces(face1_path, face2_paths)
        return jsonify(result), 200
    
    else:
        return jsonify({'error': 'Unsupported file combination'}), 400

if __name__ == '__main__':
    app.run(debug=True)