import React, { useState, useCallback, useRef } from "react";
import axios from "axios";
import { toast } from "sonner";
import { UploadCloud, Image, Youtube, Link, Plus, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function ImageUploader({ projectId, onImagesUploaded, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState("upload");
  const [youtubeUrls, setYoutubeUrls] = useState([""]);
  const [isYoutubeDialogOpen, setIsYoutubeDialogOpen] = useState(false);
  const [isProcessingYoutube, setIsProcessingYoutube] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled || !projectId) return;

    const files = Array.from(e.dataTransfer.files).filter((file) =>
      file.type.startsWith("image/")
    );

    if (files.length === 0) {
      toast.error("Please drop image files only");
      return;
    }

    await uploadFiles(files);
  }, [disabled, projectId]);

  const handleFileSelect = async (e) => {
    if (disabled || !projectId) return;

    const files = Array.from(e.target.files);
    if (files.length > 0) {
      await uploadFiles(files);
    }
    
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const uploadFiles = async (files) => {
    setIsUploading(true);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("files", file);
      });

      const response = await axios.post(
        `${API}/projects/${projectId}/images`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data.uploaded > 0) {
        toast.success(`Uploaded ${response.data.uploaded} images (filtered for architecture/landscape)`);
        onImagesUploaded(response.data.images);
      } else {
        toast.warning("No valid architecture/landscape images found. People and vehicles are filtered out.");
      }
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to upload images");
    } finally {
      setIsUploading(false);
    }
  };

  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const addYoutubeUrl = () => {
    setYoutubeUrls([...youtubeUrls, ""]);
  };

  const removeYoutubeUrl = (index) => {
    setYoutubeUrls(youtubeUrls.filter((_, i) => i !== index));
  };

  const updateYoutubeUrl = (index, value) => {
    const newUrls = [...youtubeUrls];
    newUrls[index] = value;
    setYoutubeUrls(newUrls);
  };

  const handleYoutubeSubmit = async () => {
    const validUrls = youtubeUrls.filter(
      (url) => url.trim() && (url.includes("youtube.com") || url.includes("youtu.be"))
    );

    if (validUrls.length === 0) {
      toast.error("Please enter at least one valid YouTube URL");
      return;
    }

    setIsProcessingYoutube(true);

    try {
      const response = await axios.post(`${API}/projects/${projectId}/youtube`, {
        urls: validUrls,
        fps: 6,
      });

      toast.success(
        `Started extracting frames from ${response.data.jobs.length} video(s) at 6 FPS. Architecture/landscape frames will be kept.`
      );
      setIsYoutubeDialogOpen(false);
      setYoutubeUrls([""]);

      // Start polling for YouTube job status
      pollYoutubeStatus();
    } catch (error) {
      console.error("YouTube processing error:", error);
      toast.error("Failed to start YouTube processing");
    } finally {
      setIsProcessingYoutube(false);
    }
  };

  const pollYoutubeStatus = async () => {
    if (!projectId) return;

    const checkStatus = async () => {
      try {
        const response = await axios.get(`${API}/projects/${projectId}/youtube-status`);
        const jobs = response.data;
        
        const activeJobs = jobs.filter(
          (job) => !["completed", "failed"].includes(job.status)
        );

        if (activeJobs.length > 0) {
          // Show progress for active jobs
          activeJobs.forEach((job) => {
            if (job.status === "classifying") {
              toast.info(
                `Processing: ${job.frames_accepted} frames accepted, ${job.frames_rejected} filtered out`,
                { id: `youtube-${job.id}` }
              );
            }
          });
          
          // Continue polling
          setTimeout(checkStatus, 2000);
        } else {
          // All jobs completed
          const completedJobs = jobs.filter((job) => job.status === "completed");
          if (completedJobs.length > 0) {
            const totalAccepted = completedJobs.reduce(
              (sum, job) => sum + (job.frames_accepted || 0),
              0
            );
            const totalRejected = completedJobs.reduce(
              (sum, job) => sum + (job.frames_rejected || 0),
              0
            );
            
            toast.success(
              `YouTube extraction complete! ${totalAccepted} frames added (${totalRejected} filtered out)`
            );
            
            // Refresh images list
            const imagesResponse = await axios.get(`${API}/projects/${projectId}/images`);
            onImagesUploaded(imagesResponse.data);
          }
        }
      } catch (error) {
        console.error("Error polling YouTube status:", error);
      }
    };

    setTimeout(checkStatus, 2000);
  };

  return (
    <div className="space-y-4">
      {/* Upload Zone */}
      <div
        className={`upload-dropzone ${isDragging ? "active" : ""} ${
          disabled ? "opacity-50 cursor-not-allowed" : ""
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        data-testid="image-upload-zone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled}
        />

        {isUploading ? (
          <div className="flex flex-col items-center">
            <Loader2 className="w-12 h-12 text-[var(--copper-base)] animate-spin mb-4" />
            <p className="upload-text">Uploading & Classifying...</p>
          </div>
        ) : (
          <>
            <UploadCloud className="upload-icon" />
            <p className="upload-text">
              {isDragging ? "Drop images here" : "Drag & Drop Images"}
            </p>
            <p className="upload-subtext">
              or click to browse • JPG, PNG, WEBP
            </p>
            <p className="upload-subtext mt-2">
              <Image className="inline w-3 h-3 mr-1" />
              10-50 images recommended
            </p>
          </>
        )}
      </div>

      {/* YouTube Video Option */}
      <Dialog open={isYoutubeDialogOpen} onOpenChange={setIsYoutubeDialogOpen}>
        <DialogTrigger asChild>
          <Button
            variant="outline"
            className="w-full border-[var(--surface-variant)] bg-transparent hover:bg-[var(--surface-variant)] hover:border-[var(--copper-base)] text-[var(--on-surface)]"
            disabled={disabled}
            data-testid="youtube-upload-btn"
          >
            <Youtube className="w-4 h-4 mr-2 text-red-500" />
            Extract from YouTube Videos
          </Button>
        </DialogTrigger>
        <DialogContent className="bg-[var(--surface)] border-[var(--surface-variant)] text-[var(--on-surface)] max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-[var(--on-surface)]">
              <Youtube className="w-5 h-5 text-red-500" />
              Add YouTube Videos
            </DialogTitle>
            <DialogDescription className="text-[var(--outline)]">
              Extract frames at 6 FPS. Only architecture and landscape frames are kept.
              People and vehicles are automatically filtered out.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3 py-4">
            {youtubeUrls.map((url, index) => (
              <div key={index} className="flex gap-2">
                <Input
                  value={url}
                  onChange={(e) => updateYoutubeUrl(index, e.target.value)}
                  placeholder="https://youtube.com/watch?v=..."
                  className="flex-1 bg-[var(--secondary)] border-[var(--surface-variant)] text-[var(--on-surface)] placeholder:text-[var(--outline)]"
                  data-testid={`youtube-url-input-${index}`}
                />
                {youtubeUrls.length > 1 && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeYoutubeUrl(index)}
                    className="text-[var(--outline)] hover:text-[var(--crimson)]"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
            ))}

            <Button
              variant="ghost"
              size="sm"
              onClick={addYoutubeUrl}
              className="text-[var(--copper-base)] hover:text-[var(--copper-highlight)]"
              data-testid="add-youtube-url-btn"
            >
              <Plus className="w-4 h-4 mr-1" />
              Add Another Video
            </Button>
          </div>

          <div className="bg-[var(--secondary)] rounded-lg p-3 text-xs text-[var(--outline)]">
            <p className="font-medium text-[var(--on-surface)] mb-1">Filtering Info:</p>
            <ul className="space-y-1">
              <li>• Frames with people or faces are automatically removed</li>
              <li>• Vehicles (cars, trucks, etc.) are filtered out</li>
              <li>• Only architecture, buildings, and landscapes are kept</li>
              <li>• Uses on-device NPU/GPU classification when available</li>
            </ul>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsYoutubeDialogOpen(false)}
              className="border-[var(--surface-variant)]"
            >
              Cancel
            </Button>
            <Button
              onClick={handleYoutubeSubmit}
              disabled={isProcessingYoutube}
              className="bg-[var(--crimson)] hover:bg-[var(--crimson-dark)] text-white"
              data-testid="youtube-submit-btn"
            >
              {isProcessingYoutube ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Youtube className="w-4 h-4 mr-2" />
                  Extract Frames
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Info Text */}
      <p className="text-[0.65rem] text-[var(--outline)] text-center">
        All images are analyzed to keep only architecture & landscape content
      </p>
    </div>
  );
}
