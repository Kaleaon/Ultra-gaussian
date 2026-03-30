import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Box, Cpu, Settings, Activity, Youtube, CheckCircle2, Loader2, XCircle, Triangle } from "lucide-react";
import Header from "@/components/Header";
import ImageUploader from "@/components/ImageUploader";
import ProcessingPanel from "@/components/ProcessingPanel";
import GaussianViewer from "@/components/GaussianViewer";
import MeshViewer from "@/components/MeshViewer";
import ExportPanel from "@/components/ExportPanel";
import SettingsPanel from "@/components/SettingsPanel";
import NPUStatusWidget from "@/components/NPUStatusWidget";
import { Progress } from "@/components/ui/progress";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Dashboard() {
  const [project, setProject] = useState(null);
  const [images, setImages] = useState([]);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [meshData, setMeshData] = useState(null);
  const [deviceCapabilities, setDeviceCapabilities] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState({
    quality: "high",
    resolution: 1024,
    iterations: 30000,
    sh_degree: 3,
    renderer: "gaussian",  // "gaussian" or "triangle"
  });
  const [youtubeJobs, setYoutubeJobs] = useState([]);

  // Initialize project on mount
  useEffect(() => {
    initializeProject();
    checkDeviceCapabilities();
  }, []);

  // Poll processing status when active
  useEffect(() => {
    if (!project || !processingStatus) return;
    if (processingStatus.status === "completed" || processingStatus.status === "failed") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API}/projects/${project.id}/processing-status`);
        setProcessingStatus(response.data);
        
        if (response.data.status === "completed") {
          toast.success("3D Model generation complete!");
          fetchModelData();
          fetchMeshData();
        } else if (response.data.status === "failed") {
          toast.error("Processing failed: " + response.data.error_message);
        }
      } catch (error) {
        console.error("Error polling status:", error);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [project, processingStatus]);

  // Poll YouTube jobs status
  useEffect(() => {
    if (!project) return;

    const pollYoutubeJobs = async () => {
      try {
        const response = await axios.get(`${API}/projects/${project.id}/youtube-status`);
        setYoutubeJobs(response.data);
        
        const activeJobs = response.data.filter(
          (job) => !["completed", "failed"].includes(job.status)
        );
        
        if (activeJobs.length > 0) {
          // Continue polling
          setTimeout(pollYoutubeJobs, 2000);
        } else if (response.data.length > 0) {
          // Refresh images when all jobs complete
          const imagesResponse = await axios.get(`${API}/projects/${project.id}/images`);
          setImages(imagesResponse.data);
          // Update project
          const projectResponse = await axios.get(`${API}/projects/${project.id}`);
          setProject(projectResponse.data);
        }
      } catch (error) {
        console.error("Error polling YouTube jobs:", error);
      }
    };

    // Initial check
    pollYoutubeJobs();
  }, [project?.id]);

  const initializeProject = async () => {
    try {
      // Check for existing projects
      const response = await axios.get(`${API}/projects`);
      if (response.data.length > 0) {
        const latestProject = response.data[0];
        setProject(latestProject);
        setSettings(latestProject.settings || settings);
        
        // Fetch images for this project
        const imagesResponse = await axios.get(`${API}/projects/${latestProject.id}/images`);
        setImages(imagesResponse.data);
        
        // Check processing status
        const statusResponse = await axios.get(`${API}/projects/${latestProject.id}/processing-status`);
        if (statusResponse.data.status !== "no_job") {
          setProcessingStatus(statusResponse.data);
          if (statusResponse.data.status === "completed") {
            fetchModelData(latestProject.id);
            fetchMeshData(latestProject.id);
          }
        }
      } else {
        // Create new project
        const newProject = await axios.post(`${API}/projects`, {
          name: "New 3D Model",
          description: "Gaussian Splatting reconstruction",
        });
        setProject(newProject.data);
      }
    } catch (error) {
      console.error("Error initializing project:", error);
      // Create new project as fallback
      try {
        const newProject = await axios.post(`${API}/projects`, {
          name: "New 3D Model",
          description: "Gaussian Splatting reconstruction",
        });
        setProject(newProject.data);
      } catch (createError) {
        toast.error("Failed to initialize project");
      }
    }
  };

  const checkDeviceCapabilities = async () => {
    // Check browser capabilities for WebNN/WebGPU
    const capabilities = {
      webnn: false,
      webgpu: false,
      backend: "cpu",
    };

    // Check WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          capabilities.webgpu = true;
          capabilities.backend = "gpu";
          const info = await adapter.requestAdapterInfo?.();
          capabilities.gpuInfo = info;
        }
      } catch (e) {
        console.log("WebGPU not available:", e);
      }
    }

    // Check WebNN
    if ("ml" in navigator) {
      try {
        // Try to create a context for NPU
        const context = await navigator.ml.createContext({ deviceType: "npu" });
        if (context) {
          capabilities.webnn = true;
          capabilities.backend = "npu";
          capabilities.npuAvailable = true;
        }
      } catch (e) {
        // Try GPU fallback for WebNN
        try {
          const gpuContext = await navigator.ml.createContext({ deviceType: "gpu" });
          if (gpuContext) {
            capabilities.webnn = true;
          }
        } catch (e2) {
          console.log("WebNN not available:", e2);
        }
      }
    }

    setDeviceCapabilities(capabilities);
  };

  const handleImagesUploaded = (newImages) => {
    setImages((prev) => [...prev, ...newImages]);
    // Refresh project to get updated image count
    if (project) {
      axios.get(`${API}/projects/${project.id}`).then((res) => {
        setProject(res.data);
      });
    }
  };

  const handleRemoveImage = async (imageId) => {
    setImages((prev) => prev.filter((img) => img.id !== imageId));
  };

  const handleStartProcessing = async () => {
    if (!project || images.length < 3) {
      toast.error("Please upload at least 3 images to start processing");
      return;
    }

    setIsLoading(true);
    try {
      // Update settings first
      await axios.patch(`${API}/projects/${project.id}/settings`, settings);
      
      // Start processing
      const response = await axios.post(`${API}/projects/${project.id}/process`);
      setProcessingStatus({ status: "queued", progress: 0, current_step: "Initializing..." });
      toast.success("Processing started!");
    } catch (error) {
      console.error("Error starting processing:", error);
      toast.error(error.response?.data?.detail || "Failed to start processing");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchModelData = async (projectId) => {
    const id = projectId || project?.id;
    if (!id) return;

    try {
      const response = await axios.get(`${API}/projects/${id}/model`);
      setModelData(response.data);
    } catch (error) {
      console.error("Error fetching model:", error);
    }
  };

  const fetchMeshData = async (projectId) => {
    const id = projectId || project?.id;
    if (!id) return;

    try {
      const response = await axios.get(`${API}/projects/${id}/mesh`);
      setMeshData(response.data);
    } catch (error) {
      console.error("Error fetching mesh:", error);
    }
  };

  const handleExport = async (format) => {
    if (!project) return;

    try {
      const response = await axios.get(`${API}/projects/${project.id}/export/${format}`, {
        responseType: "blob",
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `${project.name || "model"}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch (error) {
      console.error("Export error:", error);
      toast.error("Export failed");
    }
  };

  const handleSettingsChange = (newSettings) => {
    setSettings(newSettings);
  };

  const handleNewProject = async () => {
    try {
      const newProject = await axios.post(`${API}/projects`, {
        name: "New 3D Model",
        description: "Gaussian Splatting reconstruction",
      });
      setProject(newProject.data);
      setImages([]);
      setProcessingStatus(null);
      setModelData(null);
      toast.success("New project created");
    } catch (error) {
      toast.error("Failed to create new project");
    }
  };

  const isProcessing = processingStatus && 
    ["queued", "preprocessing", "training", "postprocessing"].includes(processingStatus.status);
  const isCompleted = processingStatus?.status === "completed";

  return (
    <>
      <Header 
        projectName={project?.name || "Instant3D"} 
        onNewProject={handleNewProject}
      />
      
      <main className="dashboard-grid">
        {/* Left Panel - Upload & Processing */}
        <div className="flex flex-col gap-4">
          <NPUStatusWidget 
            capabilities={deviceCapabilities} 
            isProcessing={isProcessing}
          />
          
          <div className="control-panel flex-1">
            <div className="panel-header">
              <h2 className="panel-title">
                <Box className="w-4 h-4" />
                Source Images
              </h2>
              <span className="text-xs text-[var(--outline)]">
                {images.length} uploaded
              </span>
            </div>
            <div className="panel-content">
              <ImageUploader
                projectId={project?.id}
                onImagesUploaded={handleImagesUploaded}
                disabled={isProcessing}
              />
              
              {images.length > 0 && (
                <div className="image-grid">
                  {images.slice(0, 9).map((img) => (
                    <div key={img.id} className="image-thumb" title={img.classification || "image"}>
                      <div className="w-full h-full bg-[var(--surface-variant)] flex items-center justify-center relative overflow-hidden">
                        {/* Thumbnail image */}
                        <img
                          src={`${BACKEND_URL}/api/images/${img.id}/thumbnail`}
                          alt={img.filename}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                        <div className="w-full h-full items-center justify-center hidden absolute inset-0 bg-[var(--surface-variant)]">
                          <Box className="w-5 h-5 text-[var(--outline)]" />
                        </div>
                        {(img.source === "youtube" || img.source === "video") && (
                          <Youtube className="absolute bottom-1 right-1 w-3 h-3 text-red-500 drop-shadow-md" />
                        )}
                        {img.classification && (
                          <span className="absolute top-1 left-1 text-[0.5rem] px-1 bg-[var(--surface)]/80 backdrop-blur-sm rounded text-[var(--copper-base)]">
                            {img.classification.slice(0, 4)}
                          </span>
                        )}
                      </div>
                      <button
                        className="remove-btn"
                        onClick={() => handleRemoveImage(img.id)}
                        disabled={isProcessing}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  {images.length > 9 && (
                    <div className="image-thumb flex items-center justify-center text-xs text-[var(--outline)]">
                      +{images.length - 9}
                    </div>
                  )}
                </div>
              )}

              {/* YouTube Jobs Status */}
              {youtubeJobs.length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-xs text-[var(--outline)] uppercase tracking-wider">YouTube Extractions</p>
                  {youtubeJobs.slice(-3).map((job) => (
                    <div
                      key={job.id}
                      className="flex items-center gap-2 p-2 bg-[var(--secondary)] rounded-lg text-xs"
                      data-testid={`youtube-job-${job.id}`}
                    >
                      {job.status === "completed" ? (
                        <CheckCircle2 className="w-3 h-3 text-green-500 flex-shrink-0" />
                      ) : job.status === "failed" ? (
                        <XCircle className="w-3 h-3 text-[var(--crimson)] flex-shrink-0" />
                      ) : (
                        <Loader2 className="w-3 h-3 text-[var(--copper-base)] animate-spin flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-[var(--on-surface)] truncate">{job.current_step}</p>
                        {job.status !== "completed" && job.status !== "failed" && (
                          <Progress value={job.progress} className="h-1 mt-1" />
                        )}
                        {job.status === "completed" && (
                          <p className="text-[var(--outline)]">
                            {job.frames_accepted} frames • {job.frames_rejected} filtered
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <ProcessingPanel
            processingStatus={processingStatus}
            imageCount={images.length}
            onStartProcessing={handleStartProcessing}
            isLoading={isLoading}
            isProcessing={isProcessing}
          />
        </div>

        {/* Center - 3D Viewer */}
        <div className="viewer-section">
          {settings.renderer === "triangle" ? (
            <MeshViewer
              meshData={meshData}
              isProcessing={isProcessing}
            />
          ) : (
            <GaussianViewer
              modelData={modelData}
              isProcessing={isProcessing}
              processingStatus={processingStatus}
            />
          )}
          
          {/* Renderer indicator */}
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-[var(--surface)]/80 backdrop-blur-sm px-3 py-1.5 rounded-lg">
            {settings.renderer === "triangle" ? (
              <>
                <Triangle className="w-3 h-3 text-[var(--copper-base)]" />
                <span className="text-xs text-[var(--outline)]">Triangle Splatting</span>
              </>
            ) : (
              <>
                <Box className="w-3 h-3 text-[var(--crimson)]" />
                <span className="text-xs text-[var(--outline)]">Gaussian Splatting</span>
              </>
            )}
          </div>

          {/* Load Demo Preview */}
          {!isProcessing && !modelData && !meshData && project && (
            <button
              className="absolute bottom-16 left-1/2 -translate-x-1/2 px-4 py-2 bg-[var(--surface)]/80 backdrop-blur-sm border border-[var(--surface-variant)] rounded-lg text-xs text-[var(--copper-base)] hover:bg-[var(--surface)] transition-colors"
              onClick={() => {
                if (settings.renderer === "triangle") {
                  fetchMeshData(project.id);
                } else {
                  fetchModelData(project.id);
                }
              }}
              data-testid="load-preview-btn"
            >
              Load Demo Preview
            </button>
          )}
        </div>

        {/* Right Panel - Settings & Export */}
        <div className="flex flex-col gap-4 settings-panel">
          <SettingsPanel
            settings={settings}
            onSettingsChange={handleSettingsChange}
            disabled={isProcessing}
          />
          
          <ExportPanel
            onExport={handleExport}
            disabled={!isCompleted}
            renderer={settings.renderer}
          />
        </div>
      </main>
    </>
  );
}
