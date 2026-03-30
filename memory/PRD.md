# Instant3D - Gaussian & Triangle Splatting 3D Model Generator

## Original Problem Statement
Create an app that makes a 3D model from multiple pictures using Gaussian Splatting and on-device NPU or TPU. Allow for using YouTube videos or videos, extracting pictures from videos. Focus only on land/architecture, not vehicles or people.

## User Requirements
- Direct access to device NPU/TPU (specifically Google Pixel TPU support)
- Viewer and export options for the 3D models
- Theming via Ktheme (Obsidian Crimson style)
- Extract frames from YouTube/Web/Local videos at 6 FPS
- Local/On-device image classification (MobileNet ONNX via WebNN) to filter for land/architecture
- Camera path recording in the viewer
- Support both standard 3D Gaussian Splatting and Triangle Splatting
- People in frames should be masked (inpainted) rather than discarding entire frames

## Architecture
```
Frontend (React + Tailwind + Shadcn)
├── WebNN Classifier (MobileNetV2 via ONNX Runtime Web)
│   └── /models/mobilenetv2-12.onnx + imagenet_labels.json
├── GaussianViewer (Canvas 2D, camera path recording)
├── MeshViewer (Canvas 2D, triangle rendering with wireframe)
└── Dashboard (renderer switching, upload, export)

Backend (FastAPI + MongoDB)
├── Person Masking (MediaPipe SelfieSegmentation + OpenCV inpainting)
├── Frame Classification (architecture/landscape/scene detection)
├── Gaussian Splatting Pipeline (SfM + training - SIMULATED)
├── Triangle Splatting Pipeline (Delaunay + training - SIMULATED)
├── Video Processing (yt-dlp + OpenCV @ 6 FPS)
└── Export (PLY, GLTF, OBJ, OFF)
```

## Features Implemented

### Backend
- [x] Person masking via MediaPipe SelfieSegmentation (inpaint over people, don't discard frames)
- [x] Frame classification (architecture/landscape/scene/filtered)
- [x] SfM pipeline (ORB features, essential matrix, triangulation) - SIMULATED
- [x] Gaussian Splat training with INRIA methodology - SIMULATED
- [x] Triangle Splat training (Delaunay triangulation) - SIMULATED
- [x] Image thumbnail generation
- [x] YouTube/Web/Local video frame extraction at 6 FPS
- [x] Export to PLY, GLTF, OBJ, OFF formats
- [x] Demo data generation for mesh and splat previews

### Frontend
- [x] Obsidian Crimson theme
- [x] Gaussian/Triangle renderer selector
- [x] GaussianViewer with camera path recording & playback
- [x] MeshViewer with wireframe, faces, grid toggles
- [x] "Load Demo Preview" button for quick testing
- [x] Image upload (drag & drop) with thumbnails
- [x] Video upload (File, URL, YouTube tabs)
- [x] NPU/TPU status widget with Pixel TPU Setup Guide
- [x] WebNN classifier (MobileNetV2 ONNX, local model)
- [x] Export panel (format changes based on renderer)
- [x] Settings panel (quality, resolution, iterations, SH degree)

## Key API Endpoints
- `POST /api/projects` - Create project
- `POST /api/projects/{id}/images` - Upload images (with person masking)
- `POST /api/projects/{id}/video-upload` - Upload local video
- `POST /api/projects/{id}/youtube` - YouTube frame extraction
- `POST /api/projects/{id}/web-video` - Generic web video extraction
- `POST /api/projects/{id}/process` - Start 3D reconstruction
- `GET /api/projects/{id}/model` - Get Gaussian splat data
- `GET /api/projects/{id}/mesh` - Get Triangle mesh data
- `GET /api/projects/{id}/export/{format}` - Export (ply/gltf/obj/off)

## Known Limitations
- **MOCKED Backend Pipelines**: Gaussian & Triangle Splatting training returns simulated/demo data. `pycolmap` could not be installed (`ERROR: No matching distribution found`). Real 3D reconstruction requires a working SfM library.
- WebNN/NPU requires Chrome with WebNN flag enabled on a supported device (Pixel 6+)

## Upcoming Tasks (P1)
- Implement actual SfM pipeline (alternative to pycolmap)
- Real differentiable rendering for training

## Future Tasks
- VR/AR viewing mode
- Multi-project management UI
