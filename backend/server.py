from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import uuid
from datetime import datetime, timezone
import json
import asyncio
import shutil
import base64

# Import Gaussian Splatting pipeline
from gaussian_splatting import GaussianSplatPipeline, generate_thumbnail
# Import Triangle Splatting
from triangle_splatting import TriangleSplatTrainer, convert_gaussians_to_triangles, export_to_off, export_to_obj

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
THUMBNAILS_DIR = ROOT_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

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
    renderer: str = "gaussian"  # "gaussian" or "triangle"

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


async def stream_upload_to_path(upload_file: UploadFile, destination: Path, max_bytes: int) -> int:
    """Stream an UploadFile to disk with byte limit enforcement."""
    bytes_written = 0

    try:
        with open(destination, "wb") as output_file:
            while True:
                chunk = await upload_file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(status_code=413, detail="Payload Too Large")

                output_file.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink()
        raise
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    return bytes_written

# === PERSON MASKING & CLASSIFICATION ===

# MediaPipe-based person segmentation for inpainting
_selfie_segmenter = None

def get_selfie_segmenter():
    """Lazy-init MediaPipe SelfieSegmentation (lightweight, CPU-based)."""
    global _selfie_segmenter
    if _selfie_segmenter is None:
        try:
            import mediapipe as mp
            _selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            logging.info("MediaPipe SelfieSegmentation initialized")
        except Exception as e:
            logging.warning(f"MediaPipe init failed: {e}")
    return _selfie_segmenter


def mask_people_in_frame(frame_path: str) -> dict:
    """
    Detect people in frame using MediaPipe SelfieSegmentation.
    Instead of discarding the frame, inpaint over detected person regions
    so architecture/landscape content is preserved.

    Returns:
        dict with keys: had_people (bool), person_coverage (float 0-1),
              masked_path (str - path to inpainted image, same as input if no people)
    """
    import cv2
    import numpy as np

    img = cv2.imread(frame_path)
    if img is None:
        return {"had_people": False, "person_coverage": 0.0, "masked_path": frame_path}

    segmenter = get_selfie_segmenter()
    if segmenter is None:
        # Fallback: no masking available, keep frame as-is
        return {"had_people": False, "person_coverage": 0.0, "masked_path": frame_path}

    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = segmenter.process(img_rgb)
    mask = results.segmentation_mask  # float [0,1], person=high

    # Threshold to binary mask
    person_mask = (mask > 0.5).astype(np.uint8) * 255
    person_coverage = np.sum(person_mask > 0) / (height * width)

    if person_coverage < 0.01:
        # Negligible person pixels, keep original
        return {"had_people": False, "person_coverage": float(person_coverage), "masked_path": frame_path}

    if person_coverage > 0.85:
        # Frame is almost entirely a person (selfie etc.) - discard
        return {"had_people": True, "person_coverage": float(person_coverage), "masked_path": None}

    # Dilate mask slightly for cleaner inpainting edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    person_mask_dilated = cv2.dilate(person_mask, kernel, iterations=1)

    # Inpaint over person regions using Navier-Stokes (better edge preservation)
    inpainted = cv2.inpaint(img, person_mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_NS)
    cv2.imwrite(frame_path, inpainted)

    logging.info(f"Masked people in {frame_path} (coverage: {person_coverage:.1%})")
    return {"had_people": True, "person_coverage": float(person_coverage), "masked_path": frame_path}


def classify_frame(frame_path: str) -> dict:
    """
    Classify a frame for architecture/landscape content.
    If people are detected, they are inpainted out (masked) rather than
    the frame being discarded entirely.
    """
    import cv2
    import numpy as np

    try:
        # Step 1: Mask out any people in the frame
        mask_result = mask_people_in_frame(frame_path)

        if mask_result["masked_path"] is None:
            # Frame was >85% person (e.g. selfie) - discard
            return {
                "classification": "filtered",
                "confidence": 0.9,
                "reason": f"Frame is {mask_result['person_coverage']:.0%} person (selfie/portrait)",
                "people_masked": True,
                "person_coverage": mask_result["person_coverage"],
            }

        # Step 2: Classify the (possibly inpainted) image
        img = cv2.imread(frame_path)
        if img is None:
            return {"classification": "error", "confidence": 0.0, "reason": "Cannot read image"}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        # Edge detection for architectural features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        # Line detection for buildings
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0

        # Sky detection (upper portion)
        upper_third = img[: height // 3, :, :]
        upper_hsv = cv2.cvtColor(upper_third, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(upper_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        sky_percentage = np.sum(blue_mask > 0) / (height // 3 * width)

        # Green detection for landscapes
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_percentage = np.sum(green_mask > 0) / (height * width)

        # Architecture score
        architecture_score = (
            (edge_density * 2) + (min(line_count / 100, 1.0)) + (1 - green_percentage) * 0.5 + (sky_percentage * 0.3)
        ) / 3

        # Landscape score
        landscape_score = ((sky_percentage * 1.5) + (green_percentage * 1.5) + (1 - edge_density) * 0.5) / 3

        base = {
            "people_masked": mask_result["had_people"],
            "person_coverage": mask_result["person_coverage"],
        }

        if architecture_score > 0.4 and architecture_score > landscape_score:
            return {
                **base,
                "classification": "architecture",
                "confidence": min(architecture_score, 0.95),
                "reason": "Detected architectural features",
                "features": {"edges": round(edge_density, 3), "lines": line_count},
            }
        elif landscape_score > 0.3:
            return {
                **base,
                "classification": "landscape",
                "confidence": min(landscape_score, 0.95),
                "reason": "Detected landscape features",
                "features": {"sky_pct": round(sky_percentage, 3), "green_pct": round(green_percentage, 3)},
            }
        else:
            return {
                **base,
                "classification": "scene",
                "confidence": 0.5,
                "reason": "General scene",
                "features": {"arch_score": round(architecture_score, 3), "land_score": round(landscape_score, 3)},
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
        
        # Update status for classification + person masking
        await db.video_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "classifying",
                "current_step": "Classifying frames & masking people...",
                "progress": 60,
                "frames_extracted": extracted_count
            }}
        )
        
        # Classify frames (people are masked/inpainted, not discarded)
        frame_files = sorted(frames_dir.glob("*.jpg"))
        masked_count = 0
        for i, frame_file in enumerate(frame_files):
            result = classify_frame(str(frame_file))
            
            if result["classification"] not in ["filtered", "error"]:
                people_masked = result.get("people_masked", False)
                if people_masked:
                    masked_count += 1
                accepted_frames.append({
                    "path": str(frame_file),
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "people_masked": people_masked,
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
                step_detail = f"Classified {i + 1}/{extracted_count} frames"
                if masked_count:
                    step_detail += f" ({masked_count} people masked)"
                await db.video_jobs.update_one(
                    {"id": job_id},
                    {"$set": {
                        "progress": progress,
                        "frames_accepted": len(accepted_frames),
                        "frames_rejected": rejected_count,
                        "current_step": step_detail
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
                "current_step": f"Complete ({masked_count} frames had people masked)" if masked_count else "Complete",
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
    
    allowed_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        file_ext = (Path(file.filename).suffix if file.filename else ".jpg").lower()
        if file.filename and file_ext not in allowed_image_exts:
            continue

        file_id = str(uuid.uuid4())
        file_path = project_dir / f"{file_id}{file_ext}"

        await stream_upload_to_path(file, file_path, MAX_IMAGE_BYTES)
        
        # Classify uploaded image (masks people instead of discarding)
        classification_result = classify_frame(str(file_path))
        
        # Only discard if >85% person (selfie) or error
        if classification_result["classification"] not in ["filtered", "error"]:
            # Generate thumbnail
            thumbnail_data = generate_thumbnail(str(file_path))
            thumbnail_path = None
            if thumbnail_data:
                thumb_dir = THUMBNAILS_DIR / project_id
                thumb_dir.mkdir(exist_ok=True)
                thumbnail_path = thumb_dir / f"{file_id}_thumb.jpg"
                with open(thumbnail_path, 'wb') as tf:
                    tf.write(thumbnail_data)
            
            image = ImageUpload(
                project_id=project_id,
                filename=file.filename or f"{file_id}{file_ext}",
                file_path=str(file_path),
                source="upload",
                classification=classification_result["classification"]
            )
            doc = image.model_dump()
            doc["thumbnail_path"] = str(thumbnail_path) if thumbnail_path else None
            await db.images.insert_one(doc)
            uploaded.append({
                "id": image.id,
                "filename": image.filename,
                "classification": classification_result["classification"],
                "has_thumbnail": thumbnail_path is not None
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

# Thumbnail endpoint
@api_router.get("/images/{image_id}/thumbnail")
async def get_image_thumbnail(image_id: str):
    """Get thumbnail for an image"""
    image = await db.images.find_one({"id": image_id}, {"_id": 0})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    thumbnail_path = image.get("thumbnail_path")
    if thumbnail_path and Path(thumbnail_path).exists():
        return FileResponse(thumbnail_path, media_type="image/jpeg")
    
    # Generate thumbnail on the fly if not exists
    file_path = image.get("file_path")
    if file_path and Path(file_path).exists():
        thumbnail_data = generate_thumbnail(file_path)
        if thumbnail_data:
            return Response(content=thumbnail_data, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Thumbnail not available")

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
    allowed_video_exts = {".mp4", ".webm", ".mov", ".avi", ".mpeg", ".mpg"}
    file_ext = (Path(file.filename).suffix if file.filename else ".mp4").lower()
    if file.content_type not in allowed_types or (file.filename and file_ext not in allowed_video_exts):
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: MP4, WebM, MOV, AVI, MPEG")
    
    # Save video file
    job_id = str(uuid.uuid4())
    video_dir = VIDEOS_DIR / job_id
    video_dir.mkdir(exist_ok=True)
    
    video_path = video_dir / f"video{file_ext}"

    await stream_upload_to_path(file, video_path, MAX_VIDEO_BYTES)
    
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
async def run_gaussian_splatting(job_id: str, project_id: str, settings: dict = None):
    """Run actual Gaussian Splatting pipeline with SfM and training"""
    
    try:
        # Get project images directory
        project_images_dir = UPLOAD_DIR / project_id
        frames_dir = FRAMES_DIR / project_id
        
        # Use whichever has more images
        if frames_dir.exists() and len(list(frames_dir.glob("**/*.jpg"))) > len(list(project_images_dir.glob("*"))):
            image_dir = frames_dir
        else:
            image_dir = project_images_dir
        
        output_dir = MODELS_DIR / project_id
        
        # Progress callback
        async def progress_callback(msg: str, progress: float):
            status = "preprocessing" if progress < 30 else "training" if progress < 90 else "postprocessing"
            await db.processing_jobs.update_one(
                {"id": job_id},
                {"$set": {"status": status, "progress": progress, "current_step": msg}}
            )
            await db.projects.update_one(
                {"id": project_id},
                {"$set": {"processing_progress": progress, "status": "processing"}}
            )
        
        await db.processing_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "preprocessing", "started_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        # Run the pipeline
        pipeline = GaussianSplatPipeline(str(image_dir), str(output_dir))
        splats = await pipeline.run(progress_callback=progress_callback, settings=settings)
        
        # Complete
        await db.processing_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "completed",
                "progress": 100,
                "current_step": f"Complete - {len(splats)} Gaussians",
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
        
        logger.info(f"Gaussian Splatting complete for project {project_id}: {len(splats)} splats")
        
    except Exception as e:
        logger.error(f"Gaussian Splatting failed for project {project_id}: {str(e)}")
        await db.processing_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "failed",
                "current_step": "Failed",
                "error_message": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": "failed"}}
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
    
    # Use real Gaussian Splatting pipeline
    settings = project.get("settings", {})
    background_tasks.add_task(run_gaussian_splatting, job.id, project_id, settings)
    
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
    
    # Try to load real trained model
    model_path = MODELS_DIR / project_id / "model.json"
    if model_path.exists():
        with open(model_path) as f:
            model_data = json.load(f)
            return {
                "format": model_data.get("format", "gaussian_splat"),
                "training": model_data.get("training", "differentiable_rendering"),
                "num_splats": model_data.get("num_splats", len(model_data.get("splats", []))),
                "data": model_data.get("splats", []),
                "project_id": project_id,
            }
    
    # Fallback to demo data
    demo_splat_data = generate_demo_splat_data()
    return {"format": "splat", "data": demo_splat_data, "project_id": project_id}

@api_router.get("/projects/{project_id}/export/{format}")
async def export_model(project_id: str, format: str):
    if format not in ["ply", "gltf", "obj", "off"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Use: ply, gltf, obj, off")
    
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Model not ready for export")
    
    export_path = MODELS_DIR / f"{project_id}_model.{format}"
    
    # Try to load real model data for proper export
    model_path = MODELS_DIR / project_id / "model.json"
    
    if format == "off":
        content = generate_off_mesh(model_path if model_path.exists() else None)
    elif format == "ply":
        content = generate_ply_from_model(model_path) if model_path.exists() else generate_demo_ply()
    elif format == "obj":
        content = generate_obj_from_model(model_path) if model_path.exists() else generate_demo_obj()
    else:
        content = generate_demo_gltf()
    
    with open(export_path, "w") as f:
        f.write(content)
    
    return FileResponse(
        export_path,
        filename=f"{project.get('name', 'model')}.{format}",
        media_type="application/octet-stream"
    )

# Mesh/Triangle export endpoint
@api_router.get("/projects/{project_id}/mesh")
async def get_mesh(project_id: str):
    """Get mesh data for Triangle Splatting preview"""
    project = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Try to load triangle mesh
    mesh_path = MODELS_DIR / project_id / "triangles.json"
    if mesh_path.exists():
        with open(mesh_path) as f:
            mesh_data = json.load(f)
            return {"format": "triangle", "triangles": mesh_data.get("triangles", []), "project_id": project_id}
    
    # Generate demo triangles from Gaussian splats
    model_path = MODELS_DIR / project_id / "model.json"
    if model_path.exists():
        with open(model_path) as f:
            model_data = json.load(f)
            splats = model_data.get("splats", [])
            triangles = convert_splats_to_triangles(splats)
            return {"format": "triangle", "triangles": triangles, "project_id": project_id}
    
    # Fallback to demo mesh
    demo_triangles = generate_demo_triangles()
    return {"format": "triangle", "triangles": demo_triangles, "project_id": project_id}

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


def generate_ply_from_model(model_path):
    """Generate PLY from real trained model data."""
    with open(model_path) as f:
        data = json.load(f)
    splats = data.get("splats", [])
    if not splats:
        return generate_demo_ply()
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(splats)}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "end_header",
    ]
    for s in splats:
        p = s["position"]
        sc = s.get("scale", [0.01, 0.01, 0.01])
        c = s.get("color", [128, 128, 128])
        o = s.get("opacity", 0.8)
        lines.append(f"{p[0]} {p[1]} {p[2]} {sc[0]} {sc[1]} {sc[2]} {c[0]} {c[1]} {c[2]} {o}")
    return "\n".join(lines) + "\n"


def generate_obj_from_model(model_path):
    """Generate OBJ from real trained model data (point cloud as vertices)."""
    with open(model_path) as f:
        data = json.load(f)
    splats = data.get("splats", [])
    if not splats:
        return generate_demo_obj()
    lines = ["# Gaussian Splat Export", f"# Generated by Instant3D — {len(splats)} splats"]
    for s in splats:
        p = s["position"]
        c = s.get("color", [128, 128, 128])
        lines.append(f"v {p[0]} {p[1]} {p[2]} {c[0]/255:.4f} {c[1]/255:.4f} {c[2]/255:.4f}")
    return "\n".join(lines) + "\n"

def generate_off_mesh(model_path=None):
    """Generate OFF mesh content, optionally from a trained model."""
    import random
    import math
    
    if model_path and Path(model_path).exists():
        with open(model_path) as f:
            data = json.load(f)
            splats = data.get("splats", [])
            if splats:
                lines = ["COFF\n"]
                num_tris = len(splats)
                lines.append(f"{num_tris * 3} {num_tris} 0\n")
                for s in splats:
                    pos = s["position"]
                    scale = s.get("scale", [0.01, 0.01, 0.01])
                    size = sum(scale) / len(scale) * 2
                    for k in range(3):
                        angle = k * 2 * math.pi / 3
                        vx = pos[0] + size * math.cos(angle)
                        vy = pos[1] + size * math.sin(angle)
                        vz = pos[2]
                        lines.append(f"{vx} {vy} {vz}\n")
                for i, s in enumerate(splats):
                    c = s.get("color", [128, 128, 128])
                    a = int(s.get("opacity", 0.8) * 255)
                    base = i * 3
                    lines.append(f"3 {base} {base+1} {base+2} {c[0]} {c[1]} {c[2]} {a}\n")
                return "".join(lines)
    
    # Generate demo OFF mesh
    num_triangles = 200
    lines = ["COFF\n"]
    lines.append(f"{num_triangles * 3} {num_triangles} 0\n")
    for i in range(num_triangles):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        r = 0.8 + random.gauss(0, 0.15)
        cx = r * math.sin(phi) * math.cos(theta)
        cy = r * math.sin(phi) * math.sin(theta)
        cz = r * math.cos(phi)
        size = 0.03 + random.uniform(0, 0.04)
        for k in range(3):
            angle = k * 2 * math.pi / 3 + random.uniform(0, 0.3)
            vx = cx + size * math.cos(angle)
            vy = cy + size * math.sin(angle)
            vz = cz
            lines.append(f"{vx} {vy} {vz}\n")
    for i in range(num_triangles):
        cr = int(128 + 127 * math.sin(i * 0.1))
        cg = int(128 + 127 * math.cos(i * 0.15))
        cb = int(128 + 127 * math.sin(i * 0.2 + 1))
        base = i * 3
        lines.append(f"3 {base} {base+1} {base+2} {cr} {cg} {cb} 200\n")
    return "".join(lines)

def generate_demo_triangles():
    """Generate demo triangle data for mesh preview."""
    import random
    import math
    
    triangles = []
    num_triangles = 300
    for i in range(num_triangles):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        r = 0.8 + random.gauss(0, 0.15)
        cx = r * math.sin(phi) * math.cos(theta)
        cy = r * math.sin(phi) * math.sin(theta)
        cz = r * math.cos(phi)
        size = 0.02 + random.uniform(0, 0.03)
        verts = []
        for k in range(3):
            angle = k * 2 * math.pi / 3
            verts.append([cx + size * math.cos(angle), cy + size * math.sin(angle), cz + random.gauss(0, 0.005)])
        triangles.append({
            "vertices": verts,
            "color": [int(128 + 127 * math.sin(theta)), int(128 + 127 * math.cos(phi)), int(128 + 127 * math.sin(theta + phi))],
            "opacity": random.uniform(0.6, 1.0),
        })
    return triangles

def convert_splats_to_triangles(splats):
    """Convert splat dicts to triangle dicts for mesh preview."""
    import math
    
    triangles = []
    for s in splats:
        pos = s["position"]
        scale = s.get("scale", [0.01, 0.01, 0.01])
        color = s.get("color", [128, 128, 128])
        opacity = s.get("opacity", 0.8)
        size = sum(scale) / len(scale) * 2
        verts = []
        for k in range(3):
            angle = k * 2 * math.pi / 3
            verts.append([pos[0] + size * math.cos(angle), pos[1] + size * math.sin(angle), pos[2]])
        triangles.append({
            "vertices": verts,
            "color": color,
            "opacity": opacity,
        })
    return triangles

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
