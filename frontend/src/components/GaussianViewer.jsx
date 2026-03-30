import React, { useEffect, useRef, useState } from "react";
import { Box, RotateCcw, ZoomIn, ZoomOut, Move3D, Grid3X3, Eye } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function GaussianViewer({ modelData, isProcessing, processingStatus }) {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [showGrid, setShowGrid] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const animationRef = useRef(null);

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
  }, [modelData, rotation, zoom, showGrid, showPoints, isProcessing]);

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
        </TooltipProvider>
      </div>

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
