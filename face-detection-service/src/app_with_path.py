from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from flask_cors import CORS
import logging
from deepface import DeepFace
from constants import ALLOWED_EXTENSIONS,FOLDER_NAME,HAARCASCADE_FRONTALFACE_XML
# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), FOLDER_NAME))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Check if the file extension is allowed.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_extracted_images(pdf_path, images, output_folder):
    """
    Save extracted images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        images (list): List of extracted images.
        output_folder (str): Folder to save the extracted images.

    Returns:
        list: Paths to the saved images.
    """
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    for image_index, img in enumerate(images):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_path = os.path.join(output_folder, f"extracted_image_{image_index}.{image_ext}")
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        image_paths.append(image_path)
    pdf_document.close()
    return image_paths

def extract_image_from_video(video_path, output_path):
    """
    Extract the first frame from a video file.

    Args:
        video_path (str): Path to the video file.
        output_path (str): Path to save the extracted frame.

    Returns:
        numpy.ndarray: Extracted image.
    """
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    if success:
        cv2.imwrite(output_path, image)
        video.release()
        return image
    video.release()
    return None

def detect_faces(image_path):
    """
    Detect faces in an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: List of detected face images.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAARCASCADE_FRONTALFACE_XML)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return face_images

def is_image_clear(image_path, threshold=100.0):
    """
    Check if an image is clear based on the variance of the Laplacian.

    Args:
        image_path (str): Path to the image file.
        threshold (float): Threshold for the Laplacian variance.

    Returns:
        bool: True if the image is clear, False otherwise.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def preprocess_image(image):
    """
    Preprocess the image for better face recognition.

    Args:
        image (numpy.ndarray): Image to preprocess.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    normalized_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Resize the image to a fixed size
    resized_image = cv2.resize(normalized_image, (224, 224))
    # Equalize histogram to improve contrast
    equalized = cv2.equalizeHist(resized_image)
    # Denoise image
    denoised = cv2.fastNlMeansDenoising(equalized, h=10)
    
    return denoised

def compare_faces(face1_path, face2_path, threshold=0.52, model_name='Facenet512', distance_metric='cosine'):
    """
    Compare two faces to determine if they are the same.

    Args:
        face1_path (str): Path to the first face image.
        face2_path (str): Path to the second face image.
        threshold (float): Threshold for the distance metric to consider faces the same.
        model_name (str): The model to use for face verification.
        distance_metric (str): The distance metric to use for comparison.

    Returns:
        dict: A dictionary containing the distance and whether the faces are the same.
    """
    try:
        result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, model_name=model_name, distance_metric=distance_metric)
        score = result['distance']
        is_match = score < threshold

        response = {
            'distance': score,
            'accuracy':1 - score,
            'faces_are_same': True if is_match else False
        }
        return response
    except Exception as e:
        logging.error(f"Error comparing faces: {e}")
        return {'error': 'Failed to compare faces'}

def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of extracted images.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            images += page.get_images(full=True)
        return images
    except fitz.FileDataError as e:
        logging.error(f"Error extracting images from PDF: {e}")
        return None

@app.route('/process-media', methods=['POST'])
def process_media():
    """
    Process media files for face comparison.

    This endpoint expects JSON input with the paths to the PDF and video files.
    It extracts images from the PDF and the video, detects faces, and compares the faces.

    Returns:
        JSON: A JSON response with the comparison result.
    """
    data = request.json
    pdf_path = data.get('pdf_path')
    video_path = data.get('video_path')
    
    if not pdf_path or not video_path:
        return jsonify({'error': 'Both pdf_path and video_path are required'}), 400
    
    if not os.path.exists(pdf_path) or not os.path.exists(video_path):
        return jsonify({'error': 'One or both of the paths do not exist'}), 400

    if not allowed_file(pdf_path) or not allowed_file(video_path):
        return jsonify({'error': 'Unsupported file type'}), 400

    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_images')
    os.makedirs(output_folder, exist_ok=True)
    
    if pdf_path.lower().endswith('.pdf') and video_path.lower().endswith(('.mp4', '.webm')):
        images = extract_images_from_pdf(pdf_path)
        if images is None or len(images) == 0:
            return jsonify({'error': 'No images found in the PDF file'}), 400
        
        image_paths = save_extracted_images(pdf_path, images, output_folder)
        
        clear_image_paths = [path for path in image_paths if is_image_clear(path)]
        if not clear_image_paths:
            return jsonify({'error': 'No clear images found in the PDF file. Please upload a clearer document.'}), 400

        video_image_path = os.path.join(output_folder, 'extracted_video_image.png')
        video_image = extract_image_from_video(video_path, video_image_path)
        
        if video_image is None:
            return jsonify({'error': 'Failed to extract image from video'}), 500
        
        faces_from_pdf = []
        for image_path in clear_image_paths:
            faces_from_pdf.extend(detect_faces(image_path))
        
        if len(faces_from_pdf) == 0:
            return jsonify({'error': 'No faces found in the PDF file'}), 400
        
        faces_from_video = detect_faces(video_image_path)
        
        if len(faces_from_video) == 0:
            return jsonify({'error': 'No faces found in the video'}), 400
        
        face1_path = os.path.join(output_folder, 'face1.png')
        face2_path = os.path.join(output_folder, 'face2.png')
        cv2.imwrite(face1_path, preprocess_image(faces_from_pdf[0]))
        cv2.imwrite(face2_path, preprocess_image(faces_from_video[0]))
        
        result = compare_faces(face1_path, face2_path)
        return jsonify(result), 200
    
    elif pdf_path.lower().endswith(('.png', '.jpg', '.jpeg')) and video_path.lower().endswith(('.mp4', '.webm')):
        image_path = pdf_path
        video_image_path = os.path.join(output_folder, 'extracted_video_image.png')
        video_image = extract_image_from_video(video_path, video_image_path)
        
        if video_image is None:
            return jsonify({'error': 'Failed to extract image from video'}), 500
        
        faces_from_image = detect_faces(image_path)
        
        if len(faces_from_image) == 0:
            return jsonify({'error': 'No faces found in the uploaded image'}), 400
        
        faces_from_video = detect_faces(video_image_path)
        
        if len(faces_from_video) == 0:
            return jsonify({'error': 'No faces found in the video'}), 400
        
        face1_path = os.path.join(output_folder, 'face1.png')
        face2_path = os.path.join(output_folder, 'face2.png')
        cv2.imwrite(face1_path, preprocess_image(faces_from_image[0]))
        cv2.imwrite(face2_path, preprocess_image(faces_from_video[0]))
        
        result = compare_faces(face1_path, face2_path)
        return jsonify(result), 200
    
    else:
        return jsonify({'error': 'Unsupported file combination'}), 400

if __name__ == '__main__':
    app.run(debug=True)
