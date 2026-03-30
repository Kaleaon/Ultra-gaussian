# Instant3D - Gaussian Splatting 3D Model Generator

## Original Problem Statement
Create an app that makes a 3D model from multiple pictures using Gaussian Splatting and on-device NPU or TPU. Allow for using YouTube videos or videos, extracting pictures from videos. Focus only on land/architecture, not vehicles or people.

## User Personas
1. **Architects/Designers** - Need quick 3D reconstructions of buildings
2. **Real Estate Professionals** - Create 3D tours from video walkthroughs
3. **Urban Planners** - Document and model cityscapes
4. **Hobbyists** - Create 3D models of landmarks

## Core Requirements
- Multi-image upload for 3D reconstruction
- YouTube video frame extraction at 6 FPS
- Direct video file upload support
- Architecture/landscape classification (filter out people/vehicles)
- Interactive 3D Gaussian Splat viewer
- Export to PLY, GLTF, OBJ formats
- WebNN/WebGPU for on-device acceleration

## What's Been Implemented (Jan 30, 2026)

### Backend (FastAPI + MongoDB)
- [x] Project management CRUD APIs
- [x] Multi-image upload with classification filtering
- [x] YouTube video URL processing (yt-dlp)
- [x] Direct video file upload endpoint
- [x] Frame extraction at configurable FPS
- [x] OpenCV-based frame classification:
  - Edge detection for architectural features
  - Haar cascade face detection
  - Color analysis (sky, green areas, skin tones)
  - Line detection for building structures
- [x] Processing job simulation
- [x] Model export (PLY, GLTF, OBJ)
- [x] WebNN/WebGPU capability detection endpoint

### Frontend (React + Tailwind + Shadcn)
- [x] Ktheme Obsidian Crimson design system
- [x] Drag & drop image uploader
- [x] YouTube URL input dialog (multiple URLs)
- [x] Direct video upload dialog
- [x] Interactive 3D canvas viewer
- [x] Processing progress visualization
- [x] NPU/WebGPU status widget
- [x] Settings panel (quality, resolution, iterations, SH degree)
- [x] Export controls for multiple formats
- [x] YouTube job status display

## Architecture
```
Frontend (React) <-> Backend (FastAPI) <-> MongoDB
     |                    |
     v                    v
  WebGPU/WebNN       OpenCV + yt-dlp
  (3D rendering)    (frame extraction)
```

## Prioritized Backlog

### P0 (Critical)
- [ ] Real Gaussian Splatting processing (currently simulated)
- [ ] Actual NPU/TPU inference for classification

### P1 (High)
- [ ] Image preview thumbnails
- [ ] Multiple project management
- [ ] Better frame quality selection

### P2 (Nice to have)
- [ ] 3D model editing tools
- [ ] Camera path export
- [ ] VR/AR viewing mode
- [ ] Real-time collaborative editing

## Next Tasks
1. Implement actual Gaussian Splatting processing pipeline
2. Add image thumbnail previews in the grid
3. Integrate real ML model for NPU-based classification
4. Add camera controls tutorial overlay

## Technical Notes
- YouTube downloads may be blocked by anti-bot measures; direct video upload recommended
- Classification uses OpenCV heuristics (not ML) for fast on-device processing
- 3D viewer uses Canvas 2D API; could upgrade to WebGPU for better performance
