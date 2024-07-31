ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'mp4', 'webm'}

FOLDER_NAME = 'upload_images'

HAARCASCADE_FRONTALFACE_XML = 'haarcascade_frontalface_default.xml'

HAARCASCADE_EYE_XML =  "haarcascade_eye.xml"

THRESHOLD_FOR_IS_IMAGE_CLEAR = 100.0

THRESHOLD_FOR_COMPARE_FACES = 0.52

MODEL_NAME_FOR_COMPARE_FACES = 'Facenet512'

DISTANCE_METRIC_FOR_COMPARE_FACES = 'cosine'

EXTRACTED_IMAGES_FOLDER = 'extracted_images'

PDF_EXTENSIONS = '.pdf'

VIDEO_EXTENSIONS = ('.mp4', '.webm')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

FACE1_IMAGE = 'face1.png'

FACE2_IMAGE = 'face2.png'

EXTRACTED_VIDEO_IMAGE = 'extracted_video_image.png'

# MinIO configuration
MINIO_HOST = 'dev.qritrim.com'
MINIO_PORT = 443  # Use 443 for HTTPS, 80 for HTTP
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET_KEY = 'minio123'
MINIO_BUCKET = 'minioutil'
USE_HTTPS = True  # Set to False if not using HTTPS
PROCESSED_FOLDER = 'processed_videos/'