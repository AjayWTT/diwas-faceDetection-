from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from flask_cors import CORS
import logging
from deepface import DeepFace
from config import (ALLOWED_EXTENSIONS, FOLDER_NAME, THRESHOLD_FOR_IS_IMAGE_CLEAR,
                    THRESHOLD_FOR_COMPARE_FACES, MODEL_NAME_FOR_COMPARE_FACES,
                    DISTANCE_METRIC_FOR_COMPARE_FACES, EXTRACTED_IMAGES_FOLDER,
                    PDF_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
                    FACE1_IMAGE, FACE2_IMAGE, EXTRACTED_VIDEO_IMAGE)
from mtcnn import MTCNN
import base64
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', FOLDER_NAME))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detector = MTCNN()

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

def extract_best_frame_from_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    best_frame = None
    best_score = -1

    while True:
        success, frame = video.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var > best_score:
            best_score = laplacian_var
            best_frame = frame
    
    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        video.release()
        return best_frame
    
    video.release()
    return None

def detect_faces(image_path):
    image = cv2.imread(image_path)
    detections = detector.detect_faces(image)
    face_images = []

    for detection in detections:
        x, y, width, height = detection['box']
        face = image[y:y+height, x:x+width]

        face_images.append(face)
    return face_images

def is_image_clear(image_path, threshold=THRESHOLD_FOR_IS_IMAGE_CLEAR):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def preprocess_image(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    resized_image = cv2.resize(normalized_image, (224, 224))
    return resized_image

def load_gan_model():
    """
    Load the pre-trained GAN model for age transformation.

    Returns:
        model: The loaded GAN model.
    """
    # Load the model from a file (Replace with the actual path and loading code)
    model_path = 'path_to_stargan_v2_model'  # Update this path to your GAN model
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def age_invariant_preprocessing(image_path, output_path):
    """
    Preprocess and transform a face image to a common age range using a GAN model.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the transformed image.
    """
    # Load the GAN model
    gan_model = load_gan_model()

    # Read the image
    image = cv2.imread(image_path)

    # Preprocess the image (resize and normalize)
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Age/de-age the face to a common age range using the GAN model
    # Adjust the output as needed based on the GAN's output format
    aged_image = gan_model.predict(preprocessed_image)[0]

    # Rescale to [0, 255] and convert back to BGR
    aged_image = (aged_image + 1.0) * 127.5
    aged_image = np.clip(aged_image, 0, 255).astype(np.uint8)
    aged_image = cv2.cvtColor(aged_image, cv2.COLOR_RGB2BGR)

    # Save the aged image
    cv2.imwrite(output_path, aged_image)

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
        video_image = extract_best_frame_from_video(file2_path, video_image_path)
        
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
        face2_path = os.path.join(output_folder, FACE2_IMAGE)
        age_invariant_preprocessing(faces_from_pdf[0], face1_path)
        age_invariant_preprocessing(faces_from_video[0], face2_path)
        
        result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, model_name=MODEL_NAME_FOR_COMPARE_FACES, distance_metric=DISTANCE_METRIC_FOR_COMPARE_FACES, align=True, detector_backend='mtcnn')
        return jsonify(result), 200
    
    elif file1.filename.lower().endswith(IMAGE_EXTENSIONS) and file2.filename.lower().endswith(VIDEO_EXTENSIONS):
        image_path = file1_path
        video_image_path = os.path.join(output_folder, EXTRACTED_VIDEO_IMAGE)
        video_image = extract_best_frame_from_video(file2_path, video_image_path)
        
        if video_image is None:
            return jsonify({'error': 'Failed to extract image from video'}), 500

        faces_from_image = detect_faces(image_path)
        
        if not faces_from_image:
            return jsonify({'error': 'No faces found in the uploaded image'}), 400
        
        faces_from_video = detect_faces(video_image_path)
        
        if not faces_from_video:
            return jsonify({'error': 'No faces found in the video'}), 400
        
        face1_path = os.path.join(output_folder, FACE1_IMAGE)
        face2_path = os.path.join(output_folder, FACE2_IMAGE)
        age_invariant_preprocessing(faces_from_image[0], face1_path)
        age_invariant_preprocessing(faces_from_video[0], face2_path)
        
        result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, model_name=MODEL_NAME_FOR_COMPARE_FACES, distance_metric=DISTANCE_METRIC_FOR_COMPARE_FACES, align=True, detector_backend='mtcnn')
        return jsonify(result), 200
    
    else:
        return jsonify({'error': 'Unsupported file combination'}), 400

if __name__ == '__main__':
    app.run(debug=True)
