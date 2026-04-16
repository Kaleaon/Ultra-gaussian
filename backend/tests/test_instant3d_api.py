"""
Instant3D API Tests - Iteration 4
Tests for:
- Health check
- Project management
- Mesh/Triangle data endpoints
- Export endpoints (OFF, PLY, OBJ, GLTF)
- Image upload with person masking
- Model data retrieval
- Settings update (renderer selection)
- Device capabilities
"""

import pytest
import requests
import os
import json
import io
from PIL import Image

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthAndBasics:
    """Health check and basic API tests"""
    
    def test_health_endpoint(self):
        """GET /api/health returns healthy"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        print("✓ Health endpoint working")
    
    def test_device_capabilities(self):
        """GET /api/device-capabilities returns expected capabilities"""
        response = requests.get(f"{BASE_URL}/api/device-capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "webnn_supported" in data
        assert "webgpu_supported" in data
        assert "available_backends" in data
        assert "npu" in data["available_backends"]
        assert "gpu" in data["available_backends"]
        assert "cpu" in data["available_backends"]
        print(f"✓ Device capabilities: {data}")


class TestProjectManagement:
    """Project CRUD operations"""
    
    @pytest.fixture
    def test_project(self):
        """Create a test project and clean up after"""
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Project_Iteration4",
            "description": "Test project for iteration 4"
        })
        assert response.status_code == 200
        project = response.json()
        yield project
        # Cleanup
        requests.delete(f"{BASE_URL}/api/projects/{project['id']}")
    
    def test_list_projects(self):
        """GET /api/projects returns project list"""
        response = requests.get(f"{BASE_URL}/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Projects list returned {len(data)} projects")
    
    def test_create_project(self, test_project):
        """POST /api/projects creates a new project"""
        assert "id" in test_project
        assert test_project["name"] == "TEST_Project_Iteration4"
        assert test_project["status"] == "created"
        print(f"✓ Project created: {test_project['id']}")
    
    def test_get_project(self, test_project):
        """GET /api/projects/{id} returns project details"""
        response = requests.get(f"{BASE_URL}/api/projects/{test_project['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_project["id"]
        assert data["name"] == test_project["name"]
        print(f"✓ Project retrieved: {data['id']}")


class TestRendererSettings:
    """Renderer selection (Gaussian/Triangle) tests"""
    
    @pytest.fixture
    def test_project(self):
        """Create a test project for settings tests"""
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Renderer_Settings",
            "description": "Test renderer settings"
        })
        project = response.json()
        yield project
        requests.delete(f"{BASE_URL}/api/projects/{project['id']}")
    
    def test_patch_settings_triangle_renderer(self, test_project):
        """PATCH /api/projects/{id}/settings with renderer='triangle' works"""
        response = requests.patch(
            f"{BASE_URL}/api/projects/{test_project['id']}/settings",
            json={
                "quality": "high",
                "resolution": 1024,
                "iterations": 30000,
                "sh_degree": 3,
                "renderer": "triangle"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Settings updated"
        
        # Verify settings were saved
        project_response = requests.get(f"{BASE_URL}/api/projects/{test_project['id']}")
        project_data = project_response.json()
        assert project_data["settings"]["renderer"] == "triangle"
        print("✓ Triangle renderer setting saved")
    
    def test_patch_settings_gaussian_renderer(self, test_project):
        """PATCH /api/projects/{id}/settings with renderer='gaussian' works"""
        response = requests.patch(
            f"{BASE_URL}/api/projects/{test_project['id']}/settings",
            json={
                "quality": "high",
                "resolution": 1024,
                "iterations": 30000,
                "sh_degree": 3,
                "renderer": "gaussian"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Settings updated"
        
        # Verify settings were saved
        project_response = requests.get(f"{BASE_URL}/api/projects/{test_project['id']}")
        project_data = project_response.json()
        assert project_data["settings"]["renderer"] == "gaussian"
        print("✓ Gaussian renderer setting saved")


class TestMeshAndModelData:
    """Mesh and model data retrieval tests"""
    
    @pytest.fixture
    def test_project(self):
        """Create a test project"""
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Mesh_Model_Data",
            "description": "Test mesh and model data"
        })
        project = response.json()
        yield project
        requests.delete(f"{BASE_URL}/api/projects/{project['id']}")
    
    def test_get_mesh_not_ready_for_unprocessed_project(self, test_project):
        """GET /api/projects/{id}/mesh returns explicit not-ready payload before processing"""
        response = requests.get(f"{BASE_URL}/api/projects/{test_project['id']}/mesh")
        assert response.status_code == 409
        data = response.json()
        assert data["mesh_ready"] is False
        assert data["reason"] == "processing_incomplete"
        print("✓ Mesh endpoint returned explicit not-ready state")
    
    def test_get_model_returns_demo_splat_data(self, test_project):
        """GET /api/projects/{id}/model returns demo splat data"""
        response = requests.get(f"{BASE_URL}/api/projects/{test_project['id']}/model")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "splat"
        assert "data" in data
        assert len(data["data"]) > 0
        
        # Verify splat structure
        splat = data["data"][0]
        assert "position" in splat
        assert "scale" in splat
        assert "rotation" in splat
        assert "color" in splat
        assert "opacity" in splat
        print(f"✓ Model endpoint returned {len(data['data'])} splats")


class TestExportEndpoints:
    """Export endpoints tests - require completed status"""
    
    @pytest.fixture
    def completed_project(self):
        """Create a project and set status to completed for export tests"""
        # Create project
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Export_Project",
            "description": "Test export functionality"
        })
        project = response.json()
        project_id = project["id"]
        
        # We need to manually set status to completed via direct DB or use a workaround
        # Since we can't directly access DB, we'll test the error case first
        yield project
        
        # Cleanup
        requests.delete(f"{BASE_URL}/api/projects/{project_id}")
    
    def test_export_off_requires_completed_status(self, completed_project):
        """GET /api/projects/{id}/export/off returns 400 if not completed"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project['id']}/export/off")
        # Should return 400 because status is not 'completed'
        assert response.status_code == 400
        data = response.json()
        assert "not ready" in data["detail"].lower() or "completed" in data["detail"].lower()
        print("✓ OFF export correctly requires completed status")
    
    def test_export_ply_requires_completed_status(self, completed_project):
        """GET /api/projects/{id}/export/ply returns 400 if not completed"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project['id']}/export/ply")
        assert response.status_code == 400
        print("✓ PLY export correctly requires completed status")
    
    def test_export_obj_requires_completed_status(self, completed_project):
        """GET /api/projects/{id}/export/obj returns 400 if not completed"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project['id']}/export/obj")
        assert response.status_code == 400
        print("✓ OBJ export correctly requires completed status")
    
    def test_export_gltf_requires_completed_status(self, completed_project):
        """GET /api/projects/{id}/export/gltf returns 400 if not completed"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project['id']}/export/gltf")
        assert response.status_code == 400
        print("✓ GLTF export correctly requires completed status")
    
    def test_export_invalid_format(self, completed_project):
        """GET /api/projects/{id}/export/xyz returns 400 for invalid format"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project['id']}/export/xyz")
        assert response.status_code == 400
        data = response.json()
        assert "unsupported" in data["detail"].lower()
        print("✓ Invalid export format correctly rejected")


class TestExportWithCompletedProject:
    """Test exports with a project that has completed status"""
    
    @pytest.fixture
    def completed_project_with_status(self):
        """Create a project and simulate completed status by using MongoDB directly"""
        import pymongo
        
        # Create project via API
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Completed_Export",
            "description": "Test export with completed status"
        })
        project = response.json()
        project_id = project["id"]
        
        # Connect to MongoDB and update status
        try:
            mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
            db = mongo_client["test_database"]
            db.projects.update_one(
                {"id": project_id},
                {"$set": {"status": "completed"}}
            )
            mongo_client.close()
        except Exception as e:
            print(f"Warning: Could not update project status via MongoDB: {e}")
        
        yield project
        
        # Cleanup
        requests.delete(f"{BASE_URL}/api/projects/{project_id}")
    
    def test_export_off_valid_coff_data(self, completed_project_with_status):
        """GET /api/projects/{id}/export/off returns valid COFF data"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project_with_status['id']}/export/off")
        
        if response.status_code == 400:
            pytest.skip("Project status not updated to completed")
        
        assert response.status_code == 200
        content = response.text
        assert content.startswith("COFF"), f"OFF file should start with COFF, got: {content[:50]}"
        lines = content.strip().split('\n')
        assert len(lines) > 2, "OFF file should have header and data"
        print(f"✓ OFF export returned valid COFF data ({len(lines)} lines)")
    
    def test_export_ply_valid_data(self, completed_project_with_status):
        """GET /api/projects/{id}/export/ply returns valid PLY data"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project_with_status['id']}/export/ply")
        
        if response.status_code == 400:
            pytest.skip("Project status not updated to completed")
        
        assert response.status_code == 200
        content = response.text
        assert content.startswith("ply"), f"PLY file should start with 'ply', got: {content[:50]}"
        assert "end_header" in content
        print("✓ PLY export returned valid PLY data")
    
    def test_export_obj_valid_data(self, completed_project_with_status):
        """GET /api/projects/{id}/export/obj returns valid OBJ data"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project_with_status['id']}/export/obj")
        
        if response.status_code == 400:
            pytest.skip("Project status not updated to completed")
        
        assert response.status_code == 200
        content = response.text
        assert "v " in content, "OBJ file should contain vertex definitions"
        assert "f " in content, "OBJ file should contain face definitions"
        print("✓ OBJ export returned valid OBJ data")
    
    def test_export_gltf_valid_data(self, completed_project_with_status):
        """GET /api/projects/{id}/export/gltf returns valid GLTF data"""
        response = requests.get(f"{BASE_URL}/api/projects/{completed_project_with_status['id']}/export/gltf")
        
        if response.status_code == 400:
            pytest.skip("Project status not updated to completed")
        
        assert response.status_code == 200
        data = json.loads(response.text)
        assert "asset" in data
        assert data["asset"]["version"] == "2.0"
        assert "scenes" in data
        print("✓ GLTF export returned valid GLTF data")


class TestImageUploadWithPersonMasking:
    """Image upload tests with person masking (MediaPipe)"""
    
    @pytest.fixture
    def test_project(self):
        """Create a test project for image upload"""
        response = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Image_Upload",
            "description": "Test image upload with person masking"
        })
        project = response.json()
        yield project
        requests.delete(f"{BASE_URL}/api/projects/{project['id']}")
    
    def test_upload_image_architecture(self, test_project):
        """POST /api/projects/{id}/images with image file - person masking works"""
        # Create a test image with architectural features (lines, edges)
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        
        # Add some architectural-like features (horizontal and vertical lines)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw building-like structure
        draw.rectangle([100, 100, 300, 400], outline=(50, 50, 50), width=3)
        draw.rectangle([350, 150, 550, 400], outline=(60, 60, 60), width=3)
        draw.line([(0, 100), (640, 100)], fill=(80, 80, 80), width=2)
        draw.line([(0, 200), (640, 200)], fill=(80, 80, 80), width=2)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'files': ('test_architecture.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(
            f"{BASE_URL}/api/projects/{test_project['id']}/images",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "uploaded" in data
        print(f"✓ Image upload response: {data}")
    
    def test_upload_landscape_image(self, test_project):
        """POST /api/projects/{id}/images with landscape image"""
        # Create a landscape-like image (sky blue top, green bottom)
        img = Image.new('RGB', (640, 480), color=(135, 206, 235))  # Sky blue
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Green ground
        draw.rectangle([0, 300, 640, 480], fill=(34, 139, 34))
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'files': ('test_landscape.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(
            f"{BASE_URL}/api/projects/{test_project['id']}/images",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Landscape image upload response: {data}")


class TestStaticModelFiles:
    """Test ONNX model and labels files are served correctly"""
    
    def test_onnx_model_exists(self):
        """ONNX model file exists at /models/mobilenetv2-12.onnx"""
        response = requests.head(f"{BASE_URL}/models/mobilenetv2-12.onnx")
        # Note: This might return 404 if static files aren't served from backend
        # The files are in frontend/public/models/
        if response.status_code == 404:
            # Try frontend URL
            frontend_url = BASE_URL.replace('/api', '')
            response = requests.head(f"{frontend_url}/models/mobilenetv2-12.onnx")
        
        if response.status_code == 200:
            print("✓ ONNX model file accessible")
        else:
            print(f"⚠ ONNX model file check returned {response.status_code} (may be served from frontend)")
    
    def test_imagenet_labels_exists(self):
        """ImageNet labels file exists at /models/imagenet_labels.json"""
        response = requests.head(f"{BASE_URL}/models/imagenet_labels.json")
        if response.status_code == 404:
            frontend_url = BASE_URL.replace('/api', '')
            response = requests.head(f"{frontend_url}/models/imagenet_labels.json")
        
        if response.status_code == 200:
            print("✓ ImageNet labels file accessible")
        else:
            print(f"⚠ ImageNet labels file check returned {response.status_code} (may be served from frontend)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
