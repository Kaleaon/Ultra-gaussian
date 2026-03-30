# Instant3D - Gaussian Splatting 3D Model Generator

## Original Problem Statement
Create an app that makes a 3D model from multiple pictures using Gaussian Splatting and on-device NPU or TPU. Allow for using YouTube videos or videos, extracting pictures from videos. Focus only on land/architecture, not vehicles or people. Added: Camera path recording to show where the camera was, and ability to link to any web video.

## User Personas
1. **Architects/Designers** - Need quick 3D reconstructions of buildings
2. **Real Estate Professionals** - Create 3D tours from video walkthroughs
3. **Urban Planners** - Document and model cityscapes
4. **Content Creators** - Generate 3D models from online videos

## Core Requirements
- Multi-image upload for 3D reconstruction
- Video frame extraction at 6 FPS
- Generic web video URL support (Vimeo, Twitter, direct links, etc.)
- YouTube video support
- Architecture/landscape classification (filter out people/vehicles)
- Interactive 3D Gaussian Splat viewer
- Camera path recording with playback and export
- Export to PLY, GLTF, OBJ formats

## What's Been Implemented (Jan 30, 2026)

### Backend (FastAPI + MongoDB)
- [x] Project management CRUD APIs
- [x] Multi-image upload with classification filtering
- [x] YouTube video URL processing (yt-dlp)
- [x] Generic web video URL endpoint (supports 1000+ sites)
- [x] Direct video file upload endpoint
- [x] Frame extraction at 6 FPS
- [x] OpenCV-based frame classification
- [x] Processing job simulation
- [x] Model export (PLY, GLTF, OBJ)

### Frontend (React + Tailwind + Shadcn)
- [x] Ktheme Obsidian Crimson design system
- [x] Drag & drop image uploader
- [x] Three video input options: File | URL | YouTube
- [x] Web video URL dialog (multi-URL support)
- [x] Interactive 3D canvas viewer
- [x] **Camera path recording** - records camera positions during viewing
- [x] **Camera path visualization** - shows path with start/end markers
- [x] **Camera path playback** - replay recorded camera movements
- [x] **Camera path export** - download as JSON file
- [x] Processing progress visualization
- [x] Settings panel (quality, resolution, iterations)
- [x] Export controls

### Camera Path Feature Details
- Records camera position (rotation X/Y + zoom) every 100ms
- Visualizes path with copper-colored dashed line
- Green dot = start, Red dot = end
- Numbered keyframe markers
- Playback replays exact camera movements
- Export to JSON with frame data

## Architecture
```
Frontend (React) <-> Backend (FastAPI) <-> MongoDB
     |                    |
     v                    v
  Canvas 2D           OpenCV + yt-dlp
  (3D viewer)        (frame extraction)
```

## Prioritized Backlog

### P0 (Critical)
- [ ] Real Gaussian Splatting processing (currently simulated)
- [ ] Actual NPU/TPU inference for classification

### P1 (High)
- [ ] Image preview thumbnails
- [ ] Video rendering from camera path
- [ ] Better ML model for classification

### P2 (Nice to have)
- [ ] 3D model editing tools
- [ ] VR/AR viewing mode
- [ ] Collaborative editing

## Technical Notes
- Gaussian Splatting processing is MOCKED (simulated)
- Classification uses OpenCV heuristics, not ML
- yt-dlp supports 1000+ video sites (YouTube, Vimeo, Twitter, etc.)
- Camera path stored in component state (could persist to backend)
