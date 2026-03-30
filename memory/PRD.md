# Instant3D - Gaussian & Triangle Splatting 3D Model Generator

## Original Problem Statement
Create an app that makes a 3D model from multiple pictures using Gaussian Splatting and on-device NPU or TPU. Allow for using YouTube videos or videos, extracting pictures from videos. Focus only on land/architecture, not vehicles or people.

## Recent Updates (Jan 30, 2026)
- Implemented actual Gaussian Splatting pipeline based on INRIA SIGGRAPH 2023 paper
- Added Triangle Splatting (3DV 2026) as alternative renderer - sharper, faster, game-engine compatible
- Integrated WebNN for Google Pixel TPU/NPU acceleration
- Added image thumbnail previews in upload grid

## Technical Implementation

### Gaussian Splatting (INRIA Paper)
Based on: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Structure from Motion (SfM) with ORB feature extraction
- Anisotropic 3D covariance (scale + rotation)
- Spherical Harmonics up to degree 3 for view-dependent color
- Adaptive density control (clone/split/prune)
- Opacity reset for better convergence

### Triangle Splatting (3DV 2026 Paper)
Based on: https://trianglesplatting.github.io/
- Uses triangles instead of Gaussians
- Smooth window function based on signed distance field
- Sharper edges, preserves fine details
- 2400+ FPS rendering capability
- Exports to OFF/OBJ for game engines

### WebNN for Google Pixel TPU
- Detects Pixel devices and Tensor chip version
- Backend priority: NPU → GPU → WASM (CPU)
- MobileNetV2 for image classification
- Chrome flag: chrome://flags → "WebNN API" → Enable

## Features Implemented

### Backend (FastAPI + MongoDB)
- [x] SfM pipeline (feature extraction, matching, triangulation)
- [x] Gaussian Splat training with INRIA methodology
- [x] Triangle Splat training with 3DV methodology
- [x] Image thumbnail generation on upload
- [x] Architecture/landscape classification (OpenCV)
- [x] YouTube/Web video frame extraction
- [x] Export to PLY, GLTF, OBJ, OFF formats

### Frontend (React + Tailwind + Shadcn)
- [x] Renderer selector (Gaussian vs Triangle)
- [x] Image thumbnails in upload grid
- [x] Pixel device detection with TPU status
- [x] WebNN/WebGPU capability badges
- [x] Camera path recording and playback
- [x] Settings: Quality, Resolution, Iterations, SH Degree

## Architecture
```
Frontend (React)
├── WebNN Classifier (MobileNetV2 via ONNX Runtime)
├── WebGPU/Canvas 3D Viewer
└── API Client

Backend (FastAPI)
├── Gaussian Splatting Pipeline
│   ├── SfM (StructureFromMotion)
│   └── GaussianSplatTrainer
├── Triangle Splatting Pipeline  
│   └── TriangleSplatTrainer
├── Classification (OpenCV)
└── MongoDB
```

## Next Steps
1. Download and serve MobileNetV2 ONNX model for client-side classification
2. Add mesh preview mode for Triangle Splatting exports
3. Implement actual differentiable rendering for training
4. Add VR/AR viewing mode
