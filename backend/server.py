from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import json
import asyncio
import shutil

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create directories
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
FRAMES_DIR = ROOT_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = ROOT_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# Create the main app
app = FastAPI(title="Instant3D - Gaussian Splatting API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# === MODELS ===

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class Project(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    status: str = "created"
    image_count: int = 0
    processing_progress: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_url: Optional[str] = None
    settings: dict = Field(default_factory=lambda: {
        "quality": "high",
        "resolution": 1024,
        "iterations": 30000,
        "sh_degree": 3
    })

class ProcessingSettings(BaseModel):
    quality: str = "high"
    resolution: int = 1024
    iterations: int = 30000
    sh_degree: int = 3

class ImageUpload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    filename: str
    file_path: str
    source: str = "upload"  # upload, youtube
    classification: Optional[str] = None  # architecture, landscape, filtered
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ProcessingJob(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    status: str = "queued"
    progress: float = 0.0
    current_step: str = "Initializing"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class YouTubeRequest(BaseModel):
    urls: List[str]
    fps: int = 6

class WebVideoRequest(BaseModel):
    urls: List[str]
    fps: int = 6

class VideoUploadRequest(BaseModel):
    fps: int = 6

class VideoExtractionJob(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    video_url: str
    status: str = "queued"  # queued, downloading, extracting, classifying, completed, failed
    progress: float = 0.0
    current_step: str = "Queued"
    frames_extracted: int = 0
    frames_accepted: int = 0
    frames_rejected: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

# === CLASSIFICATION HELPERS ===

# Simple keyword-based classification for architecture/landscape detection
# This runs fast and can be enhanced with actual ML models later
ARCHITECTURE_KEYWORDS = [
    'building', 'house', 'tower', 'bridge', 'church', 'castle', 'monument',
    'skyscraper', 'temple', 'palace', 'cathedral', 'dome', 'arch', 'column',
    'facade', 'roof', 'window', 'door', 'wall', 'staircase', 'balcony'
]

LANDSCAPE_KEYWORDS = [
    'mountain', 'valley', 'river', 'lake', 'forest', 'field', 'meadow',
    'cliff', 'canyon', 'beach', 'ocean', 'desert', 'hill', 'rock',
    'garden', 'park', 'tree', 'grass', 'sky', 'horizon'
]

FILTER_KEYWORDS = [
    'person', 'people', 'face', 'human', 'man', 'woman', 'child', 'crowd',
    'car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic',
    'selfie', 'portrait', 'group'
]

def classify_frame_simple(frame_path: str) -> dict:
    """
    Simple frame classification based on image analysis.
    Returns classification result with confidence.
    
    In production, this would use a lightweight MobileNet or similar model
    that can run on NPU/TPU via ONNX Runtime with WebNN backend.
    """
    import cv2
    import numpy as np
    
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return {"classification": "error", "confidence": 0.0, "reason": "Cannot read image"}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate various features
        height, width = img.shape[:2]
        
        # Edge detection for architectural features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Line detection for buildings
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Color analysis
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Sky detection (upper portion of image)
        upper_third = img[:height//3, :, :]
        upper_hsv = cv2.cvtColor(upper_third, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(upper_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        sky_percentage = np.sum(blue_mask > 0) / (height//3 * width)
        
        # Green detection for landscapes
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_percentage = np.sum(green_mask > 0) / (height * width)
        
        # Skin tone detection for people filtering
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_percentage = np.sum(skin_mask > 0) / (height * width)
        
        # Face detection using Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_faces = len(faces) > 0
        
        # Decision logic
        # Filter out images with people
        if has_faces or skin_percentage > 0.15:
            return {
                "classification": "filtered",
                "confidence": 0.85,
                "reason": "Contains people/faces",
                "features": {"faces": len(faces), "skin_pct": round(skin_percentage, 3)}
            }
        
        # Architecture detection: high edge density + many lines + less green
        architecture_score = (
            (edge_density * 2) +
            (min(line_count / 100, 1.0)) +
            (1 - green_percentage) * 0.5 +
            (sky_percentage * 0.3)
        ) / 3
        
        # Landscape detection: sky presence + green areas + lower edge density
        landscape_score = (
            (sky_percentage * 1.5) +
            (green_percentage * 1.5) +
            (1 - edge_density) * 0.5
        ) / 3
        
        if architecture_score > 0.4 and architecture_score > landscape_score:
            return {
                "classification": "architecture",
                "confidence": min(architecture_score, 0.95),
                "reason": "Detected architectural features",
                "features": {"edges": round(edge_density, 3), "lines": line_count}
            }
        elif landscape_score > 0.3:
            return {
                "classification": "landscape",
                "confidence": min(landscape_score, 0.95),
                "reason": "Detected landscape features",
                "features": {"sky_pct": round(sky_percentage, 3), "green_pct": round(green_percentage, 3)}
            }
        else:
            # Accept as generic scene if not clearly people/vehicles
            return {
                "classification": "scene",
                "confidence": 0.5,
                "reason": "General scene - architecture/landscape unclear",
                "features": {"arch_score": round(architecture_score, 3), "land_score": round(landscape_score, 3)}
            }
            
    except Exception as e:
        return {"classification": "error", "confidence": 0.0, "reason": str(e)}


# === YOUTUBE PROCESSING ===

async def download_and_extract_youtube(job_id: str, project_id: str, video_url: str, fps: int = 6):
    """Download YouTube video and extract frames at specified FPS"""
    import yt_dlp
    import cv2
    
    try:
        # Update job status
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "downloading",
                "current_step": "Downloading video...",
                "started_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        # Setup yt-dlp
        video_dir = VIDEOS_DIR / job_id
        video_dir.mkdir(exist_ok=True)
        
        ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            'outtmpl': str(video_dir / 'video.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 3,
        }
        
        # Download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'Unknown')
        
        # Find downloaded video file
        video_files = list(video_dir.glob('video.*'))
        if not video_files:
            raise Exception("Video download failed - no file found")
        
        video_path = str(video_files[0])
        
        # Continue with frame extraction
        await extract_frames_from_video(job_id, project_id, video_path, fps)
        
    except Exception as e:
        logging.error(f"YouTube extraction failed: {str(e)}")
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "failed",
                "current_step": "Failed",
                "error_message": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }}
        )


async def extract_frames_from_video(job_id: str, project_id: str, video_path: str, fps: int = 6):
    """Extract frames from a video file and classify them"""
    import cv2
    
    try:
        # Update status
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "extracting",
                "current_step": f"Extracting frames at {fps} FPS...",
                "progress": 20,
                "started_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        # Extract frames
        frames_dir = FRAMES_DIR / project_id / job_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))
        
        frame_count = 0
        extracted_count = 0
        accepted_frames = []
        rejected_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = frames_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    progress = 20 + (frame_count / max(total_frames, 1)) * 40
                    await db.video_jobs.update_one(
                        {"id": job_id},
                        {"$set": {
                            "progress": progress,
                            "frames_extracted": extracted_count,
                            "current_step": f"Extracted {extracted_count} frames..."
                        }}
                    )
            
            frame_count += 1
        
        cap.release()
        
        # Update status for classification
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "classifying",
                "current_step": "Classifying frames (filtering people/vehicles)...",
                "progress": 60,
                "frames_extracted": extracted_count
            }}
        )
        
        # Classify frames
        frame_files = sorted(frames_dir.glob("*.jpg"))
        for i, frame_file in enumerate(frame_files):
            result = classify_frame_simple(str(frame_file))
            
            if result["classification"] in ["architecture", "landscape", "scene"]:
                accepted_frames.append({
                    "path": str(frame_file),
                    "classification": result["classification"],
                    "confidence": result["confidence"]
                })
                
                image_record = ImageUpload(
                    project_id=project_id,
                    filename=frame_file.name,
                    file_path=str(frame_file),
                    source="video",
                    classification=result["classification"]
                )
                await db.images.insert_one(image_record.model_dump())
            else:
                frame_file.unlink()
                rejected_count += 1
            
            if (i + 1) % 10 == 0:
                progress = 60 + ((i + 1) / max(extracted_count, 1)) * 35
                await db.video_jobs.update_one(
                    {"id": job_id},
                    {"$set": {
                        "progress": progress,
                        "frames_accepted": len(accepted_frames),
                        "frames_rejected": rejected_count,
                        "current_step": f"Classified {i + 1}/{extracted_count} frames..."
                    }}
                )
        
        # Update project image count
        current_project = await db.projects.find_one({"id": project_id}, {"_id": 0})
        new_count = current_project.get("image_count", 0) + len(accepted_frames)
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {
                "image_count": new_count,
                "status": "uploading",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        # Cleanup video file
        video_dir = VIDEOS_DIR / job_id
        if video_dir.exists():
            shutil.rmtree(video_dir)
        
        # Complete
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "completed",
                "progress": 100,
                "current_step": "Complete",
                "frames_accepted": len(accepted_frames),
                "frames_rejected": rejected_count,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
    except Exception as e:
        logging.error(f"Video extraction failed: {str(e)}")
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "failed",
                "current_step": "Failed",
                "error_message": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }}
        )


# === ROUTES ===

@api_router.get("/")
async def root():
    return {"message": "Instant3D API - Gaussian Splatting Engine", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Project Management
@api_router.post("/projects", response_model=Project)
async def create_project(input_data: ProjectCreate):
    project = Project(name=input_data.name, description=input_data.description)
    doc = project.model_dump()
    await db.projects.insert_one(doc)
    return project

@api_router.get("/projects", response_model=List[Project])
async def list_projects():
    projects = await db.projects.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return projects

@api_router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    result = await db.projects.delete_one({"id": project_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.images.delete_many({"project_id": project_id})
    await db.processing_jobs.delete_many({"project_id": project_id})
    await db.video_jobs.delete_many({"project_id": project_id})
    return {"message": "Project deleted successfully"}

@api_router.patch("/projects/{project_id}/settings")
async def update_project_settings(project_id: str, settings: ProcessingSettings):
    result = await db.projects.update_one(
        {"id": project_id},
        {"$set": {"settings": settings.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Settings updated"}

# Image Upload
@api_router.post("/projects/{project_id}/images")
async def upload_images(project_id: str, files: List[UploadFile] = File(...)):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    uploaded = []
    project_dir = UPLOAD_DIR / project_id
    project_dir.mkdir(exist_ok=True)
    
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix if file.filename else ".jpg"
        file_path = project_dir / f"{file_id}{file_ext}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Classify uploaded image
        classification_result = classify_frame_simple(str(file_path))
        
        # Only accept architecture/landscape/scene images
        if classification_result["classification"] in ["architecture", "landscape", "scene"]:
            image = ImageUpload(
                project_id=project_id,
                filename=file.filename or f"{file_id}{file_ext}",
                file_path=str(file_path),
                source="upload",
                classification=classification_result["classification"]
            )
            await db.images.insert_one(image.model_dump())
            uploaded.append({
                "id": image.id,
                "filename": image.filename,
                "classification": classification_result["classification"]
            })
        else:
            # Delete filtered image
            file_path.unlink()
    
    new_count = project.get("image_count", 0) + len(uploaded)
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"image_count": new_count, "status": "uploading", "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    return {"uploaded": len(uploaded), "images": uploaded}

@api_router.get("/projects/{project_id}/images")
async def list_project_images(project_id: str):
    images = await db.images.find({"project_id": project_id}, {"_id": 0}).to_list(1000)
    return images

# YouTube Video Processing
@api_router.post("/projects/{project_id}/youtube")
async def add_youtube_videos(project_id: str, request: YouTubeRequest, background_tasks: BackgroundTasks):
    """Add YouTube videos to extract frames from"""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    jobs = []
    for url in request.urls:
        # Validate YouTube URL
        if not ("youtube.com" in url or "youtu.be" in url):
            continue
        
        job = VideoExtractionJob(project_id=project_id, video_url=url)
        await db.video_jobs.insert_one(job.model_dump())
        
        # Start background extraction
        background_tasks.add_task(
            download_and_extract_youtube,
            job.id,
            project_id,
            url,
            request.fps
        )
        
        jobs.append({"job_id": job.id, "url": url, "status": "queued"})
    
    return {"jobs": jobs, "fps": request.fps}

# Direct Video Upload Processing
@api_router.post("/projects/{project_id}/video-upload")
async def upload_video(project_id: str, file: UploadFile = File(...), fps: int = 6, background_tasks: BackgroundTasks = None):
    """Upload a video file directly to extract frames from"""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate file type
    allowed_types = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/mpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: MP4, WebM, MOV, AVI, MPEG")
    
    # Save video file
    job_id = str(uuid.uuid4())
    video_dir = VIDEOS_DIR / job_id
    video_dir.mkdir(exist_ok=True)
    
    file_ext = Path(file.filename).suffix if file.filename else ".mp4"
    video_path = video_dir / f"video{file_ext}"
    
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)
    
    # Create job record
    job = VideoExtractionJob(
        id=job_id,
        project_id=project_id,
        video_url=f"file://{video_path}"
    )
    await db.video_jobs.insert_one(job.model_dump())
    
    # Start background extraction
    background_tasks.add_task(
        extract_frames_from_video,
        job_id,
        project_id,
        str(video_path),
        fps
    )
    
    return {"job_id": job_id, "filename": file.filename, "status": "processing", "fps": fps}

@api_router.get("/projects/{project_id}/youtube-status")
async def get_youtube_status(project_id: str):
    """Get status of all YouTube extraction jobs for a project"""
    jobs = await db.video_jobs.find({"project_id": project_id}, {"_id": 0}).to_list(100)
    return jobs

# Generic Web Video URL Processing
@api_router.post("/projects/{project_id}/web-video")
async def add_web_videos(project_id: str, request: WebVideoRequest, background_tasks: BackgroundTasks):
    """Add any web video URLs to extract frames from (supports YouTube, Vimeo, Twitter, etc.)"""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    jobs = []
    for url in request.urls:
        url = url.strip()
        if not url:
            continue
        
        # Validate URL format
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        
        job = VideoExtractionJob(project_id=project_id, video_url=url)
        await db.video_jobs.insert_one(job.model_dump())
        
        # Start background extraction using yt-dlp (supports many sites)
        background_tasks.add_task(
            download_and_extract_youtube,
            job.id,
            project_id,
            url,
            request.fps
        )
        
        jobs.append({"job_id": job.id, "url": url, "status": "queued"})
    
    return {"jobs": jobs, "fps": request.fps}

# Processing
async def simulate_processing(job_id: str, project_id: str):
    """Simulate Gaussian Splatting processing steps"""
    
    steps = [
        ("Detecting camera poses (COLMAP)", 10),
        ("Extracting features", 25),
        ("Point cloud initialization", 35),
        ("Gaussian optimization - iteration 5000", 50),
        ("Gaussian optimization - iteration 15000", 65),
        ("Gaussian optimization - iteration 25000", 80),
        ("Densification complete", 90),
        ("Generating splat model", 95),
        ("Finalizing export", 100),
    ]
    
    await db.processing_jobs.update_one(
        {"id": job_id},
        {"$set": {"status": "preprocessing", "started_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    for step_name, progress in steps:
        await asyncio.sleep(2)
        status = "training" if progress > 30 else "preprocessing"
        if progress >= 90:
            status = "postprocessing"
        
        await db.processing_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": status, "progress": progress, "current_step": step_name}}
        )
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"processing_progress": progress, "status": "processing"}}
        )
    
    await db.processing_jobs.update_one(
        {"id": job_id},
        {"$set": {
            "status": "completed",
            "progress": 100,
            "current_step": "Complete",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {
            "status": "completed",
            "processing_progress": 100,
            "model_url": f"/api/projects/{project_id}/model"
        }}
    )

@api_router.post("/projects/{project_id}/process")
async def start_processing(project_id: str, background_tasks: BackgroundTasks):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.get("image_count", 0) < 3:
        raise HTTPException(status_code=400, detail="At least 3 images required for reconstruction")
    
    existing_job = await db.processing_jobs.find_one(
        {"project_id": project_id, "status": {"$in": ["queued", "preprocessing", "training", "postprocessing"]}},
        {"_id": 0}
    )
    if existing_job:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    job = ProcessingJob(project_id=project_id)
    await db.processing_jobs.insert_one(job.model_dump())
    
    background_tasks.add_task(simulate_processing, job.id, project_id)
    
    return {"job_id": job.id, "status": "started"}

@api_router.get("/projects/{project_id}/processing-status")
async def get_processing_status(project_id: str):
    job = await db.processing_jobs.find_one(
        {"project_id": project_id},
        {"_id": 0},
        sort=[("started_at", -1)]
    )
    if not job:
        return {"status": "no_job", "progress": 0}
    return job

# Model Export
@api_router.get("/projects/{project_id}/model")
async def get_model(project_id: str):
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    demo_splat_data = generate_demo_splat_data()
    return {"format": "splat", "data": demo_splat_data, "project_id": project_id}

@api_router.get("/projects/{project_id}/export/{format}")
async def export_model(project_id: str, format: str):
    if format not in ["ply", "gltf", "obj"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use: ply, gltf, obj")
    
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Model not ready for export")
    
    export_path = MODELS_DIR / f"{project_id}_model.{format}"
    
    if format == "ply":
        content = generate_demo_ply()
    elif format == "obj":
        content = generate_demo_obj()
    else:
        content = generate_demo_gltf()
    
    with open(export_path, "w") as f:
        f.write(content)
    
    return FileResponse(
        export_path,
        filename=f"{project.get('name', 'model')}.{format}",
        media_type="application/octet-stream"
    )

@api_router.get("/device-capabilities")
async def get_device_capabilities():
    return {
        "webnn_supported": True,
        "webgpu_supported": True,
        "available_backends": ["npu", "gpu", "cpu"],
        "recommended_backend": "npu",
        "note": "Actual capability detection happens in the browser via WebNN API"
    }

# Helper functions
def generate_demo_splat_data():
    import random
    import math
    
    splats = []
    num_points = 5000
    
    for i in range(num_points):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        r = 1.0 + random.gauss(0, 0.2)
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        color_r = int(128 + 127 * math.sin(theta))
        color_g = int(128 + 127 * math.cos(phi))
        color_b = int(128 + 127 * math.sin(theta + phi))
        
        splats.append({
            "position": [x, y, z],
            "scale": [0.01 + random.uniform(0, 0.02)] * 3,
            "rotation": [0, 0, 0, 1],
            "color": [color_r, color_g, color_b],
            "opacity": random.uniform(0.7, 1.0)
        })
    
    return splats

def generate_demo_ply():
    return """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 255 0 0
1 0 0 0 255 0
1 1 0 0 0 255
0 1 0 255 255 0
0 0 1 255 0 255
1 0 1 0 255 255
1 1 1 255 255 255
0 1 1 128 128 128
"""

def generate_demo_obj():
    return """# Gaussian Splat Export - Demo
# Generated by Instant3D
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 3 4 8 7
f 1 4 8 5
f 2 3 7 6
"""

def generate_demo_gltf():
    return json.dumps({
        "asset": {"version": "2.0", "generator": "Instant3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [],
        "bufferViews": [],
        "buffers": []
    }, indent=2)

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
