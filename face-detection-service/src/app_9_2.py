from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from flask_cors import CORS
import logging
from deepface import DeepFace
from config import (
    ALLOWED_EXTENSIONS, FOLDER_NAME, HAARCASCADE_FRONTALFACE_XML,
    THRESHOLD_FOR_IS_IMAGE_CLEAR, THRESHOLD_FOR_COMPARE_FACES,
    MODEL_NAME_FOR_COMPARE_FACES, DISTANCE_METRIC_FOR_COMPARE_FACES,
    EXTRACTED_IMAGES_FOLDER, PDF_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS,
    FACE1_IMAGE, FACE2_IMAGE, EXTRACTED_VIDEO_IMAGE, 
    MINIO_HOST, MINIO_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, USE_HTTPS
)
from minio import Minio
import subprocess

# Initialize MinIO client
try:
    minio_client = Minio(
        f"{MINIO_HOST}:{MINIO_PORT}",
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=USE_HTTPS
    )
except Exception as e:
    logging.error(f"Failed to initialize MinIO client: {e}")
    exit(1)

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

import cv2

def preprocess_image(image):

    # Convert to YUV color space and equalize histogram for better contrast
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Apply CLAHE for better contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Apply super-resolution (assuming a pre-trained model is loaded)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(r"C:\Users\WalkingTree\Desktop\New folder\diwas\face-detection-service\EDSR_x4.pb")
    sr.setModel("edsr", 4)  # Example with EDSR model with a scale factor of 4
    image = sr.upsample(image)

    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    normalized_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Resize the image to the required input size for the model
    resized_image = cv2.resize(normalized_image, (224, 224))

    return resized_image



def compare_faces(face1_path, face2_path, threshold=0.52, model_name=MODEL_NAME_FOR_COMPARE_FACES, distance_metric=DISTANCE_METRIC_FOR_COMPARE_FACES):
    try:
        result = DeepFace.verify(img1_path=face1_path, img2_path=face2_path, model_name=model_name, distance_metric='cosine', align=True, detector_backend='mtcnn')
        score = result['distance']
        is_match = score < threshold

        response = {
            'distance': score,
            'accuracy': 1 - score,
            'faces_are_same': is_match
        }
        return response
    except Exception as e:
        logging.error(f"Error comparing faces: {e}")
        return {'error': 'Failed to compare faces'}

def download_from_minio(object_name, file_path):
    try:
        minio_client.fget_object(
            MINIO_BUCKET,
            f"upload_images/{object_name}",
            file_path
        )
        logging.info(f"Downloaded {object_name} from MinIO to {file_path}")
    except Exception as e:
        logging.error(f"Failed to download {object_name} from MinIO: {e}")

@app.route('/process-media', methods=['POST'])
def process_media():
    pdf_key = request.form.get('pdf_key')
    video_key = request.form.get('video_key')
    
    if not pdf_key or not video_key:
        return jsonify({'error': 'PDF key and video key are required'}), 400
    
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf_key))
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_key))
    
    try:
        download_from_minio(pdf_key, file1_path)
        download_from_minio(video_key, file2_path)
    except Exception as e:
        return jsonify({'error': f'Failed to download files: {e}'}), 500
    
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], EXTRACTED_IMAGES_FOLDER)
    os.makedirs(output_folder, exist_ok=True)

    video_name = video_key.split('.')[0]
    converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"converted_{video_name}.mp4")
    
    try:
        ffmpeg_command = [
            'ffmpeg', '-i', file2_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast',
            '-c:a', 'aac', '-b:a', '128k', '-vsync', 'vfr', '-y', converted_video_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, ffmpeg_command)
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Failed to convert video: {e}")
        return jsonify({'error': f"Failed to convert video: {str(e)}"}), 500

    if file1_path.lower().endswith(PDF_EXTENSIONS) and file2_path.lower().endswith(VIDEO_EXTENSIONS):
        images = convert_from_path(file1_path)
        if not images:
            return jsonify({'error': 'No images found in the PDF file/Upload correct PDF file'}), 400
        
        image_paths = save_extracted_images(images, output_folder)
        
        clear_image_paths = [path for path in image_paths if is_image_clear(path)]
        if not clear_image_paths:
            return jsonify({'error': 'No clear images found in the PDF file. Please upload a clearer document.'}), 400

        video_image_path = os.path.join(output_folder, EXTRACTED_VIDEO_IMAGE)
        video_image = extract_best_frame_from_video(converted_video_path, video_image_path)
        
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
        cv2.imwrite(face1_path, preprocess_image(faces_from_pdf[0]))
        cv2.imwrite(face2_path, preprocess_image(faces_from_video[0]))
        
        result = compare_faces(face1_path, face2_path)
        return jsonify(result), 200
    
    elif file1_path.lower().endswith(IMAGE_EXTENSIONS) and file2_path.lower().endswith(VIDEO_EXTENSIONS):
        image_path = file1_path
        video_image_path = os.path.join(output_folder, EXTRACTED_VIDEO_IMAGE)
        video_image = extract_best_frame_from_video(converted_video_path, video_image_path)
        
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
        cv2.imwrite(face1_path, preprocess_image(faces_from_image[0]))
        cv2.imwrite(face2_path, preprocess_image(faces_from_video[0]))
        
        result = compare_faces(face1_path, face2_path)
        return jsonify(result), 200
    
    else:
        return jsonify({'error': 'Incorrect file formats. Please upload a PDF and a video file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
