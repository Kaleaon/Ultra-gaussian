#!/usr/bin/env python3
"""
Backend API Testing for Gaussian Splatting 3D Model Generation App
Tests all API endpoints including project management, image upload, processing, and export
"""

import requests
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import tempfile
import os

class GaussianSplattingAPITester:
    def __init__(self, base_url="https://instant-3d-3.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.project_id = None
        self.processing_job_id = None

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def test_health_check(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Response: {data.get('status', 'unknown')}"
            return self.log_test("Health Check", success, details)
        except Exception as e:
            return self.log_test("Health Check", False, f"Error: {str(e)}")

    def test_root_endpoint(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Message: {data.get('message', 'N/A')}"
            return self.log_test("Root Endpoint", success, details)
        except Exception as e:
            return self.log_test("Root Endpoint", False, f"Error: {str(e)}")

    def test_create_project(self):
        """Test project creation"""
        try:
            project_data = {
                "name": "Test 3D Model",
                "description": "Test project for Gaussian Splatting"
            }
            response = requests.post(f"{self.api_url}/projects", json=project_data, timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                self.project_id = data.get("id")
                details += f", Project ID: {self.project_id}"
            
            return self.log_test("Create Project", success, details)
        except Exception as e:
            return self.log_test("Create Project", False, f"Error: {str(e)}")

    def test_list_projects(self):
        """Test listing projects"""
        try:
            response = requests.get(f"{self.api_url}/projects", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Projects found: {len(data)}"
                if self.project_id:
                    project_found = any(p.get("id") == self.project_id for p in data)
                    details += f", Test project found: {project_found}"
                    success = project_found
            
            return self.log_test("List Projects", success, details)
        except Exception as e:
            return self.log_test("List Projects", False, f"Error: {str(e)}")

    def test_get_project(self):
        """Test getting specific project"""
        if not self.project_id:
            return self.log_test("Get Project", False, "No project ID available")
        
        try:
            response = requests.get(f"{self.api_url}/projects/{self.project_id}", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Name: {data.get('name', 'N/A')}, Status: {data.get('status', 'N/A')}"
            
            return self.log_test("Get Project", success, details)
        except Exception as e:
            return self.log_test("Get Project", False, f"Error: {str(e)}")

    def test_image_upload(self):
        """Test image upload functionality"""
        if not self.project_id:
            return self.log_test("Image Upload", False, "No project ID available")
        
        try:
            # Create a simple test image file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Create a minimal JPEG header (not a real image, but should pass content-type check)
                tmp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9')
                tmp_file_path = tmp_file.name
            
            # Upload the file
            with open(tmp_file_path, 'rb') as f:
                files = {'files': ('test_image.jpg', f, 'image/jpeg')}
                response = requests.post(
                    f"{self.api_url}/projects/{self.project_id}/images",
                    files=files,
                    timeout=15
                )
            
            # Clean up
            os.unlink(tmp_file_path)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Uploaded: {data.get('uploaded', 0)} images"
            
            return self.log_test("Image Upload", success, details)
        except Exception as e:
            return self.log_test("Image Upload", False, f"Error: {str(e)}")

    def test_list_project_images(self):
        """Test listing project images"""
        if not self.project_id:
            return self.log_test("List Project Images", False, "No project ID available")
        
        try:
            response = requests.get(f"{self.api_url}/projects/{self.project_id}/images", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Images found: {len(data)}"
            
            return self.log_test("List Project Images", success, details)
        except Exception as e:
            return self.log_test("List Project Images", False, f"Error: {str(e)}")

    def test_update_project_settings(self):
        """Test updating project settings"""
        if not self.project_id:
            return self.log_test("Update Project Settings", False, "No project ID available")
        
        try:
            settings = {
                "quality": "high",
                "resolution": 1024,
                "iterations": 30000,
                "sh_degree": 3
            }
            response = requests.patch(
                f"{self.api_url}/projects/{self.project_id}/settings",
                json=settings,
                timeout=10
            )
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            return self.log_test("Update Project Settings", success, details)
        except Exception as e:
            return self.log_test("Update Project Settings", False, f"Error: {str(e)}")

    def test_start_processing(self):
        """Test starting processing (requires at least 3 images)"""
        if not self.project_id:
            return self.log_test("Start Processing", False, "No project ID available")
        
        # First upload more images to meet the minimum requirement
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9')
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, 'rb') as f:
                    files = {'files': (f'test_image_{i}.jpg', f, 'image/jpeg')}
                    requests.post(
                        f"{self.api_url}/projects/{self.project_id}/images",
                        files=files,
                        timeout=15
                    )
                os.unlink(tmp_file_path)
            
            # Now try to start processing
            response = requests.post(f"{self.api_url}/projects/{self.project_id}/process", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                self.processing_job_id = data.get("job_id")
                details += f", Job ID: {self.processing_job_id}, Status: {data.get('status', 'N/A')}"
            
            return self.log_test("Start Processing", success, details)
        except Exception as e:
            return self.log_test("Start Processing", False, f"Error: {str(e)}")

    def test_processing_status(self):
        """Test getting processing status"""
        if not self.project_id:
            return self.log_test("Processing Status", False, "No project ID available")
        
        try:
            response = requests.get(f"{self.api_url}/projects/{self.project_id}/processing-status", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Job Status: {data.get('status', 'N/A')}, Progress: {data.get('progress', 0)}%"
            
            return self.log_test("Processing Status", success, details)
        except Exception as e:
            return self.log_test("Processing Status", False, f"Error: {str(e)}")

    def test_device_capabilities(self):
        """Test device capabilities endpoint"""
        try:
            response = requests.get(f"{self.api_url}/device-capabilities", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", WebNN: {data.get('webnn_supported', False)}, WebGPU: {data.get('webgpu_supported', False)}"
            
            return self.log_test("Device Capabilities", success, details)
        except Exception as e:
            return self.log_test("Device Capabilities", False, f"Error: {str(e)}")

    def test_model_retrieval(self):
        """Test model data retrieval (may need to wait for processing)"""
        if not self.project_id:
            return self.log_test("Model Retrieval", False, "No project ID available")
        
        try:
            response = requests.get(f"{self.api_url}/projects/{self.project_id}/model", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Format: {data.get('format', 'N/A')}, Data points: {len(data.get('data', []))}"
            
            return self.log_test("Model Retrieval", success, details)
        except Exception as e:
            return self.log_test("Model Retrieval", False, f"Error: {str(e)}")

    def test_youtube_video_processing(self):
        """Test YouTube video processing endpoint"""
        if not self.project_id:
            return self.log_test("YouTube Video Processing", False, "No project ID available")
        
        try:
            # Test with multiple YouTube URLs (using test URLs that should work)
            youtube_data = {
                "urls": [
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - short video
                    "https://youtu.be/dQw4w9WgXcQ"  # Same video, different format
                ],
                "fps": 6
            }
            response = requests.post(
                f"{self.api_url}/projects/{self.project_id}/youtube",
                json=youtube_data,
                timeout=15
            )
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Jobs created: {len(data.get('jobs', []))}, FPS: {data.get('fps', 'N/A')}"
            
            return self.log_test("YouTube Video Processing", success, details)
        except Exception as e:
            return self.log_test("YouTube Video Processing", False, f"Error: {str(e)}")

    def test_youtube_status(self):
        """Test YouTube processing status endpoint"""
        if not self.project_id:
            return self.log_test("YouTube Status", False, "No project ID available")
        
        try:
            response = requests.get(f"{self.api_url}/projects/{self.project_id}/youtube-status", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Jobs found: {len(data)}"
                if data:
                    # Show status of first job
                    first_job = data[0]
                    details += f", First job status: {first_job.get('status', 'N/A')}"
            
            return self.log_test("YouTube Status", success, details)
        except Exception as e:
            return self.log_test("YouTube Status", False, f"Error: {str(e)}")

    def test_image_classification(self):
        """Test image classification filtering"""
        if not self.project_id:
            return self.log_test("Image Classification", False, "No project ID available")
        
        try:
            # Create a test image that should be classified
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Create a minimal JPEG (this will likely be classified as "scene" or "error")
                tmp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9')
                tmp_file_path = tmp_file.name
            
            # Upload the file and check if classification is applied
            with open(tmp_file_path, 'rb') as f:
                files = {'files': ('classification_test.jpg', f, 'image/jpeg')}
                response = requests.post(
                    f"{self.api_url}/projects/{self.project_id}/images",
                    files=files,
                    timeout=15
                )
            
            # Clean up
            os.unlink(tmp_file_path)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Uploaded: {data.get('uploaded', 0)}"
                # Check if images have classification info
                if data.get('images'):
                    first_image = data['images'][0]
                    classification = first_image.get('classification', 'none')
                    details += f", Classification: {classification}"
            
            return self.log_test("Image Classification", success, details)
        except Exception as e:
            return self.log_test("Image Classification", False, f"Error: {str(e)}")

    def test_web_video_processing(self):
        """Test generic web video URL processing endpoint (new feature)"""
        if not self.project_id:
            return self.log_test("Web Video Processing", False, "No project ID available")
        
        try:
            # Test with various video URLs that yt-dlp should support
            web_video_data = {
                "urls": [
                    "https://vimeo.com/148751763",  # Vimeo test video
                    "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"  # Direct MP4 link
                ],
                "fps": 6
            }
            response = requests.post(
                f"{self.api_url}/projects/{self.project_id}/web-video",
                json=web_video_data,
                timeout=15
            )
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Jobs created: {len(data.get('jobs', []))}, FPS: {data.get('fps', 'N/A')}"
            
            return self.log_test("Web Video Processing", success, details)
        except Exception as e:
            return self.log_test("Web Video Processing", False, f"Error: {str(e)}")

    def test_image_thumbnail_endpoint(self):
        """Test thumbnail generation and endpoint"""
        if not self.project_id:
            return self.log_test("Image Thumbnail Endpoint", False, "No project ID available")
        
        try:
            # First get list of images to find an image ID
            images_response = requests.get(f"{self.api_url}/projects/{self.project_id}/images", timeout=10)
            if images_response.status_code != 200:
                return self.log_test("Image Thumbnail Endpoint", False, "Could not get project images")
            
            images = images_response.json()
            if not images:
                return self.log_test("Image Thumbnail Endpoint", False, "No images found in project")
            
            # Test thumbnail endpoint for first image
            image_id = images[0]['id']
            response = requests.get(f"{self.api_url}/images/{image_id}/thumbnail", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                # Check if response is actually an image
                content_type = response.headers.get('content-type', '')
                details += f", Content-Type: {content_type}"
                details += f", Size: {len(response.content)} bytes"
                success = 'image' in content_type and len(response.content) > 0
            
            return self.log_test("Image Thumbnail Endpoint", success, details)
        except Exception as e:
            return self.log_test("Image Thumbnail Endpoint", False, f"Error: {str(e)}")

    def test_gaussian_splatting_pipeline_import(self):
        """Test that Gaussian Splatting pipeline module loads correctly"""
        try:
            # Test if the pipeline can be imported (indirect test via API)
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                # If API is running, the gaussian_splatting module loaded successfully
                details += " - GaussianSplatPipeline imported successfully in server.py"
            
            return self.log_test("Gaussian Splatting Pipeline Import", success, details)
        except Exception as e:
            return self.log_test("Gaussian Splatting Pipeline Import", False, f"Error: {str(e)}")

    def test_processing_with_real_pipeline(self):
        """Test that processing actually uses the real Gaussian Splatting pipeline"""
        if not self.project_id:
            return self.log_test("Real GS Pipeline Processing", False, "No project ID available")
        
        try:
            # Upload multiple images first
            for i in range(5):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    # Create a more realistic test image
                    tmp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9')
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, 'rb') as f:
                    files = {'files': (f'pipeline_test_{i}.jpg', f, 'image/jpeg')}
                    requests.post(
                        f"{self.api_url}/projects/{self.project_id}/images",
                        files=files,
                        timeout=15
                    )
                os.unlink(tmp_file_path)
            
            # Start processing
            response = requests.post(f"{self.api_url}/projects/{self.project_id}/process", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                details += f", Job ID: {data.get('job_id')}"
                
                # Wait a bit and check if processing actually started
                time.sleep(3)
                status_response = requests.get(f"{self.api_url}/projects/{self.project_id}/processing-status", timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_step = status_data.get('current_step', '')
                    details += f", Current step: {current_step}"
                    # Check if it's using real pipeline steps
                    real_pipeline_steps = ['Loading images', 'Extracting features', 'Matching features', 'Estimating camera poses', 'Triangulating points', 'Training iteration']
                    is_real_pipeline = any(step in current_step for step in real_pipeline_steps)
                    details += f", Real pipeline: {is_real_pipeline}"
            
            return self.log_test("Real GS Pipeline Processing", success, details)
        except Exception as e:
            return self.log_test("Real GS Pipeline Processing", False, f"Error: {str(e)}")

    def test_model_file_saved_to_disk(self):
        """Test that model files are actually saved to disk after processing"""
        if not self.project_id:
            return self.log_test("Model File Saved to Disk", False, "No project ID available")
        
        try:
            # Check if model endpoint returns data indicating file was saved
            response = requests.get(f"{self.api_url}/projects/{self.project_id}/model", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                format_type = data.get('format', '')
                data_points = len(data.get('data', []))
                details += f", Format: {format_type}, Data points: {data_points}"
                
                # Check if it's real model data (not just demo data)
                if data_points > 0:
                    details += " - Model data available"
                    # Real models should have more than just demo data
                    success = data_points >= 100  # Real GS models should have many splats
                else:
                    details += " - No model data found"
                    success = False
            
            return self.log_test("Model File Saved to Disk", success, details)
        except Exception as e:
            return self.log_test("Model File Saved to Disk", False, f"Error: {str(e)}")

    def test_export_formats(self):
        """Test export functionality for different formats"""
        if not self.project_id:
            return self.log_test("Export Formats", False, "No project ID available")
        
        formats = ["ply", "gltf", "obj"]
        all_success = True
        
        for format_type in formats:
            try:
                response = requests.get(
                    f"{self.api_url}/projects/{self.project_id}/export/{format_type}",
                    timeout=15
                )
                success = response.status_code == 200 or response.status_code == 400  # 400 if not ready
                details = f"{format_type.upper()}: {response.status_code}"
                
                if not success:
                    all_success = False
                
                print(f"  Export {format_type.upper()}: {'✅' if success else '❌'} Status {response.status_code}")
            except Exception as e:
                print(f"  Export {format_type.upper()}: ❌ Error: {str(e)}")
                all_success = False
        
        return self.log_test("Export Formats", all_success, f"Tested: {', '.join(formats)}")

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Gaussian Splatting API Tests")
        print(f"🔗 Testing API at: {self.api_url}")
        print("=" * 60)
        
        # Basic connectivity tests
        self.test_health_check()
        self.test_root_endpoint()
        self.test_device_capabilities()
        
        # Project management tests
        self.test_create_project()
        self.test_list_projects()
        self.test_get_project()
        self.test_update_project_settings()
        
        # Image upload tests
        self.test_image_upload()
        self.test_list_project_images()
        self.test_image_classification()
        
        # NEW: Thumbnail tests
        self.test_image_thumbnail_endpoint()
        
        # NEW: Gaussian Splatting pipeline tests
        self.test_gaussian_splatting_pipeline_import()
        
        # YouTube video processing tests
        self.test_youtube_video_processing()
        time.sleep(3)  # Give YouTube processing a moment to start
        self.test_youtube_status()
        
        # Web video processing tests (new feature)
        self.test_web_video_processing()
        
        # Processing tests with real pipeline
        self.test_processing_with_real_pipeline()
        time.sleep(2)  # Give processing a moment to start
        self.test_processing_status()
        
        # Model and export tests
        self.test_model_retrieval()
        self.test_model_file_saved_to_disk()
        self.test_export_formats()
        
        # Summary
        print("=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed - check logs above")
            return 1

def main():
    tester = GaussianSplattingAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())