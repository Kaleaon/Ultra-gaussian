import React, { useEffect, useRef, useState } from "react";
import { Box, RotateCcw, ZoomIn, ZoomOut, Move3D, Grid3X3, Eye, Video, Circle, Play, Trash2, Download } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";

export default function GaussianViewer({ modelData, isProcessing, processingStatus }) {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [showGrid, setShowGrid] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const animationRef = useRef(null);
  
  // Camera path recording
  const [isRecording, setIsRecording] = useState(false);
  const [cameraPath, setCameraPath] = useState([]);
  const [isPlayingPath, setIsPlayingPath] = useState(false);
  const [showCameraPath, setShowCameraPath] = useState(true);
  const playbackRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const updateSize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };
    updateSize();
    window.addEventListener("resize", updateSize);

    // Animation loop
    const render = () => {
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const scale = Math.min(canvas.width, canvas.height) * 0.3 * zoom;

      // Draw grid
      if (showGrid) {
        ctx.strokeStyle = "#1a1a1a";
        ctx.lineWidth = 1;
        const gridSize = 50;
        for (let i = -5; i <= 5; i++) {
          const offset = i * gridSize * zoom;
          // Horizontal
          ctx.beginPath();
          ctx.moveTo(0, centerY + offset);
          ctx.lineTo(canvas.width, centerY + offset);
          ctx.stroke();
          // Vertical
          ctx.beginPath();
          ctx.moveTo(centerX + offset, 0);
          ctx.lineTo(centerX + offset, canvas.height);
          ctx.stroke();
        }
      }

      // Draw camera path
      if (showCameraPath && cameraPath.length > 1) {
        ctx.strokeStyle = "#B87333";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        
        cameraPath.forEach((point, idx) => {
          // Convert camera rotation to screen position (simplified visualization)
          const pathX = centerX + (point.y * 2);
          const pathY = centerY - (point.x * 2);
          
          if (idx === 0) {
            ctx.moveTo(pathX, pathY);
          } else {
            ctx.lineTo(pathX, pathY);
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw camera position markers
        cameraPath.forEach((point, idx) => {
          const pathX = centerX + (point.y * 2);
          const pathY = centerY - (point.x * 2);
          
          ctx.beginPath();
          ctx.arc(pathX, pathY, 4, 0, Math.PI * 2);
          ctx.fillStyle = idx === 0 ? "#22C55E" : idx === cameraPath.length - 1 ? "#DC143C" : "#B87333";
          ctx.fill();
          
          // Draw frame number
          if (idx % 5 === 0 || idx === cameraPath.length - 1) {
            ctx.fillStyle = "#8A8A8A";
            ctx.font = "10px JetBrains Mono";
            ctx.fillText(`${idx + 1}`, pathX + 6, pathY + 3);
          }
        });
      }

      // Draw splat points if we have model data
      if (modelData && modelData.data && showPoints) {
        const splats = modelData.data;
        const rotX = rotation.x * Math.PI / 180;
        const rotY = rotation.y * Math.PI / 180;

        // Sort by depth for proper rendering
        const sortedSplats = splats
          .map((splat, idx) => {
            const [x, y, z] = splat.position;
            // Apply rotation
            const cosX = Math.cos(rotX);
            const sinX = Math.sin(rotX);
            const cosY = Math.cos(rotY);
            const sinY = Math.sin(rotY);

            let rx = x;
            let ry = y * cosX - z * sinX;
            let rz = y * sinX + z * cosX;

            let fx = rx * cosY + rz * sinY;
            let fy = ry;
            let fz = -rx * sinY + rz * cosY;

            return { ...splat, fx, fy, fz, idx };
          })
          .sort((a, b) => b.fz - a.fz);

        sortedSplats.forEach((splat) => {
          const screenX = centerX + splat.fx * scale;
          const screenY = centerY - splat.fy * scale;
          const size = Math.max(2, (splat.scale[0] * scale * 50) / (1 + Math.abs(splat.fz)));

          const [r, g, b] = splat.color;
          const alpha = splat.opacity * (0.5 + 0.5 * (1 / (1 + Math.abs(splat.fz))));

          ctx.beginPath();
          ctx.arc(screenX, screenY, size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
          ctx.fill();
        });
      } else if (!isProcessing) {
        // Draw placeholder
        ctx.fillStyle = "#1a1a1a";
        ctx.font = "14px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.fillText("Upload images to generate 3D model", centerX, centerY);
      }

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", updateSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [modelData, rotation, zoom, showGrid, showPoints, isProcessing, cameraPath, showCameraPath]);

  // Camera path recording
  useEffect(() => {
    if (isRecording) {
      recordingIntervalRef.current = setInterval(() => {
        setCameraPath(prev => [...prev, { x: rotation.x, y: rotation.y, zoom, timestamp: Date.now() }]);
      }, 100); // Record position every 100ms
    } else {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    }
    
    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    };
  }, [isRecording, rotation, zoom]);

  // Camera path playback
  useEffect(() => {
    if (isPlayingPath && cameraPath.length > 0) {
      let currentIdx = 0;
      
      playbackRef.current = setInterval(() => {
        if (currentIdx < cameraPath.length) {
          const point = cameraPath[currentIdx];
          setRotation({ x: point.x, y: point.y });
          setZoom(point.zoom);
          currentIdx++;
        } else {
          setIsPlayingPath(false);
          clearInterval(playbackRef.current);
        }
      }, 100);
    } else {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    }
    
    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    };
  }, [isPlayingPath, cameraPath]);

  // Mouse handlers for rotation
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setRotation((prev) => ({
      x: prev.x + dy * 0.5,
      y: prev.y + dx * 0.5,
    }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.max(0.5, Math.min(3, prev * delta)));
  };

  const handleReset = () => {
    setRotation({ x: 0, y: 0 });
    setZoom(1);
  };

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
    } else {
      setCameraPath([]);
      setIsRecording(true);
    }
  };

  const clearPath = () => {
    setCameraPath([]);
    setIsRecording(false);
    setIsPlayingPath(false);
  };

  const exportCameraPath = () => {
    if (cameraPath.length === 0) return;
    
    const pathData = {
      version: "1.0",
      frames: cameraPath.length,
      duration_ms: cameraPath.length * 100,
      path: cameraPath.map((p, idx) => ({
        frame: idx,
        rotation_x: p.x,
        rotation_y: p.y,
        zoom: p.zoom
      }))
    };
    
    const blob = new Blob([JSON.stringify(pathData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "camera_path.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="viewer-canvas-container" data-testid="webgpu-viewer-canvas">
      <canvas
        ref={canvasRef}
        className="viewer-canvas cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Processing Overlay */}
      {isProcessing && (
        <div className="processing-overlay">
          <div className="processing-spinner mb-4" />
          <p className="text-sm text-[var(--on-surface)] mb-2">
            {processingStatus?.current_step || "Processing..."}
          </p>
          <p className="text-xs text-[var(--copper-base)]">
            {Math.round(processingStatus?.progress || 0)}% complete
          </p>
        </div>
      )}

      {/* Empty State */}
      {!modelData && !isProcessing && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <Box className="w-16 h-16 text-[var(--surface-variant)] mb-4" />
          <p className="text-lg font-medium text-[var(--on-surface)] mb-2">
            No Model Loaded
          </p>
          <p className="text-sm text-[var(--outline)]">
            Upload images and start processing
          </p>
        </div>
      )}

      {/* Floating Controls */}
      <div className="viewer-controls">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="viewer-control-btn"
                onClick={handleReset}
                data-testid="viewer-reset-btn"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Reset View</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="viewer-control-btn"
                onClick={() => setZoom((z) => Math.min(3, z * 1.2))}
                data-testid="viewer-zoom-in-btn"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Zoom In</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="viewer-control-btn"
                onClick={() => setZoom((z) => Math.max(0.5, z * 0.8))}
                data-testid="viewer-zoom-out-btn"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Zoom Out</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`viewer-control-btn ${showGrid ? "active" : ""}`}
                onClick={() => setShowGrid(!showGrid)}
                data-testid="viewer-grid-btn"
              >
                <Grid3X3 className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Toggle Grid</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`viewer-control-btn ${showPoints ? "active" : ""}`}
                onClick={() => setShowPoints(!showPoints)}
                data-testid="viewer-points-btn"
              >
                <Eye className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Toggle Points</TooltipContent>
          </Tooltip>

          <div className="w-px h-6 bg-[var(--surface-variant)] mx-1" />

          {/* Camera Path Controls */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`viewer-control-btn ${isRecording ? "active" : ""}`}
                onClick={toggleRecording}
                disabled={isPlayingPath}
                data-testid="viewer-record-btn"
              >
                <Circle className={`w-4 h-4 ${isRecording ? "text-red-500 animate-pulse" : ""}`} />
              </button>
            </TooltipTrigger>
            <TooltipContent>{isRecording ? "Stop Recording" : "Record Camera Path"}</TooltipContent>
          </Tooltip>

          {cameraPath.length > 0 && (
            <>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    className={`viewer-control-btn ${isPlayingPath ? "active" : ""}`}
                    onClick={() => setIsPlayingPath(!isPlayingPath)}
                    disabled={isRecording}
                    data-testid="viewer-play-btn"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>{isPlayingPath ? "Stop Playback" : "Play Path"}</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    className="viewer-control-btn"
                    onClick={exportCameraPath}
                    data-testid="viewer-export-path-btn"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Export Camera Path</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    className="viewer-control-btn"
                    onClick={clearPath}
                    data-testid="viewer-clear-path-btn"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Clear Path</TooltipContent>
              </Tooltip>
            </>
          )}
        </TooltipProvider>
      </div>

      {/* Recording indicator */}
      {isRecording && (
        <div className="absolute top-4 right-4 flex items-center gap-2 glass-panel px-3 py-2 rounded-lg">
          <Circle className="w-3 h-3 text-red-500 animate-pulse fill-red-500" />
          <span className="text-xs text-[var(--on-surface)]">Recording: {cameraPath.length} frames</span>
        </div>
      )}

      {/* Camera path info */}
      {cameraPath.length > 0 && !isRecording && (
        <div className="absolute top-4 right-4 glass-panel px-3 py-2 rounded-lg">
          <p className="text-xs text-[var(--copper-base)]">
            Camera Path: {cameraPath.length} keyframes ({(cameraPath.length * 0.1).toFixed(1)}s)
          </p>
        </div>
      )}

      {/* Rotation Info */}
      {modelData && (
        <div className="absolute top-4 left-4 glass-panel px-3 py-2 rounded-lg">
          <p className="text-[0.65rem] text-[var(--outline)] font-mono">
            X: {Math.round(rotation.x)}° Y: {Math.round(rotation.y)}° Z: {(zoom * 100).toFixed(0)}%
          </p>
        </div>
      )}
    </div>
  );
}
