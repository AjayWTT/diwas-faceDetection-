import unittest
import sys
import os
from io import BytesIO
from flask import Flask
from flask_testing import TestCase

# Insert the path to the src directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from app import app  # Import the Flask app from the src directory

class TestMediaProcessing(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "tests", "test_upload_images")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        return app

    def tearDown(self):
        folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    def test_process_media_pdf_and_video(self):
        pdf_path = os.path.join('tests', 'test_files', 'test_pdf.pdf')
        video_path = os.path.join('tests', 'test_files', 'test_video.mp4')
        with open(pdf_path, 'rb') as pdf, open(video_path, 'rb') as video:
            response = self.client.post('/process-media', data={
                'pdf': (BytesIO(pdf.read()), 'test_pdf.pdf'),
                'video': (BytesIO(video.read()), 'test_video.mp4')
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn('distance', response.json)
        self.assertIn('faces_are_same', response.json)

    def test_process_media_image_and_video(self):
        image_path = os.path.join('tests', 'test_files', 'test_image.jpg')
        video_path = os.path.join('tests', 'test_files', 'test_video.mp4')
        with open(image_path, 'rb') as image, open(video_path, 'rb') as video:
            response = self.client.post('/process-media', data={
                'pdf': (BytesIO(image.read()), 'test_image.jpg'),
                'video': (BytesIO(video.read()), 'test_video.mp4')
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn('distance', response.json)
        self.assertIn('faces_are_same', response.json)

    def test_process_media_missing_files(self):
        response = self.client.post('/process-media', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_process_media_unsupported_file_type(self):
        txt_path = os.path.join('tests', 'test_files', 'test_invalid.txt')
        video_path = os.path.join('tests', 'test_files', 'test_video.mp4')
        with open(txt_path, 'rb') as txt, open(video_path, 'rb') as video:
            response = self.client.post('/process-media', data={
                'pdf': (BytesIO(txt.read()), 'test_invalid.txt'),
                'video': (BytesIO(video.read()), 'test_video.mp4')
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_process_media_no_faces_in_pdf(self):
        pdf_path = os.path.join('tests', 'test_files', 'no_faces_pdf.pdf')
        video_path = os.path.join('tests', 'test_files', 'test_video.mp4')
        with open(pdf_path, 'rb') as pdf, open(video_path, 'rb') as video:
            response = self.client.post('/process-media', data={
                'pdf': (BytesIO(pdf.read()), 'no_faces_pdf.pdf'),
                'video': (BytesIO(video.read()), 'test_video.mp4')
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_process_media_no_faces_in_video(self):
        pdf_path = os.path.join('tests', 'test_files', 'test_pdf.pdf')
        video_path = os.path.join('tests', 'test_files', 'no_faces_video.mp4')
        with open(pdf_path, 'rb') as pdf, open(video_path, 'rb') as video:
            response = self.client.post('/process-media', data={
                'pdf': (BytesIO(pdf.read()), 'test_pdf.pdf'),
                'video': (BytesIO(video.read()), 'no_faces.mp4')
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_process_media_empty_file(self):
        response = self.client.post('/process-media', data={
            'pdf': (BytesIO(b''), 'empty_pdf.pdf'),
            'video': (BytesIO(b''), 'empty_video.mp4')
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

if __name__ == '__main__':
    unittest.main()
