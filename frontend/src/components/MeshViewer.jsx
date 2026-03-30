import React, { useEffect, useRef, useState } from "react";
import { Box, RotateCcw, ZoomIn, ZoomOut, Grid3X3, Eye, Layers, Download } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * MeshViewer - 3D mesh preview for Triangle Splatting exports
 * Renders triangles with wireframe and solid modes
 */
export default function MeshViewer({ meshData, isProcessing }) {
  const canvasRef = useRef(null);
  const [rotation, setRotation] = useState({ x: 30, y: 45 });
  const [zoom, setZoom] = useState(1);
  const [showWireframe, setShowWireframe] = useState(true);
  const [showFaces, setShowFaces] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const updateSize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };
    updateSize();
    window.addEventListener("resize", updateSize);

    const render = () => {
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const scale = Math.min(canvas.width, canvas.height) * 0.35 * zoom;

      // Draw grid
      if (showGrid) {
        ctx.strokeStyle = "#1a1a1a";
        ctx.lineWidth = 1;
        const gridSize = 50;
        for (let i = -5; i <= 5; i++) {
          const offset = i * gridSize * zoom;
          ctx.beginPath();
          ctx.moveTo(0, centerY + offset);
          ctx.lineTo(canvas.width, centerY + offset);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(centerX + offset, 0);
          ctx.lineTo(centerX + offset, canvas.height);
          ctx.stroke();
        }
      }

      // Draw triangles
      if (meshData && meshData.triangles) {
        const triangles = meshData.triangles;
        const rotX = (rotation.x * Math.PI) / 180;
        const rotY = (rotation.y * Math.PI) / 180;

        // Transform and sort triangles by depth
        const transformedTriangles = triangles.map((tri) => {
          const vertices = tri.vertices.map((v) => {
            // Apply rotation
            const cosX = Math.cos(rotX);
            const sinX = Math.sin(rotX);
            const cosY = Math.cos(rotY);
            const sinY = Math.sin(rotY);

            let [x, y, z] = v;

            // Rotate around X
            let ry = y * cosX - z * sinX;
            let rz = y * sinX + z * cosX;
            y = ry;
            z = rz;

            // Rotate around Y
            let rx = x * cosY + z * sinY;
            rz = -x * sinY + z * cosY;
            x = rx;

            return { x, y, z };
          });

          // Calculate centroid for depth sorting
          const centroidZ =
            (vertices[0].z + vertices[1].z + vertices[2].z) / 3;

          // Project to 2D
          const projected = vertices.map((v) => ({
            x: centerX + v.x * scale,
            y: centerY - v.y * scale,
          }));

          return {
            ...tri,
            projected,
            depth: centroidZ,
            normal: calculateNormal(vertices),
          };
        });

        // Sort by depth (painter's algorithm)
        transformedTriangles.sort((a, b) => a.depth - b.depth);

        // Draw faces
        if (showFaces) {
          transformedTriangles.forEach((tri) => {
            const { projected, color, opacity, normal } = tri;

            // Simple shading based on normal
            const lightDir = { x: 0.5, y: 0.5, z: 1 };
            const shade = Math.max(
              0.3,
              normal.x * lightDir.x + normal.y * lightDir.y + normal.z * lightDir.z
            );

            const r = Math.round((color?.[0] || 128) * shade);
            const g = Math.round((color?.[1] || 128) * shade);
            const b = Math.round((color?.[2] || 128) * shade);
            const a = opacity || 0.8;

            ctx.beginPath();
            ctx.moveTo(projected[0].x, projected[0].y);
            ctx.lineTo(projected[1].x, projected[1].y);
            ctx.lineTo(projected[2].x, projected[2].y);
            ctx.closePath();
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
            ctx.fill();
          });
        }

        // Draw wireframe
        if (showWireframe) {
          ctx.strokeStyle = showFaces ? "rgba(184, 115, 51, 0.6)" : "#B87333";
          ctx.lineWidth = showFaces ? 1 : 1.5;

          transformedTriangles.forEach((tri) => {
            const { projected } = tri;
            ctx.beginPath();
            ctx.moveTo(projected[0].x, projected[0].y);
            ctx.lineTo(projected[1].x, projected[1].y);
            ctx.lineTo(projected[2].x, projected[2].y);
            ctx.closePath();
            ctx.stroke();
          });
        }

        // Draw triangle count
        ctx.fillStyle = "#8A8A8A";
        ctx.font = "12px JetBrains Mono";
        ctx.fillText(`${triangles.length} triangles`, 10, canvas.height - 10);
      } else if (!isProcessing) {
        ctx.fillStyle = "#1a1a1a";
        ctx.font = "14px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.fillText("No mesh data - select Triangle renderer", centerX, centerY);
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
  }, [meshData, rotation, zoom, showWireframe, showFaces, showGrid, isProcessing]);

  // Calculate triangle normal
  function calculateNormal(vertices) {
    const v0 = vertices[0];
    const v1 = vertices[1];
    const v2 = vertices[2];

    const edge1 = { x: v1.x - v0.x, y: v1.y - v0.y, z: v1.z - v0.z };
    const edge2 = { x: v2.x - v0.x, y: v2.y - v0.y, z: v2.z - v0.z };

    const normal = {
      x: edge1.y * edge2.z - edge1.z * edge2.y,
      y: edge1.z * edge2.x - edge1.x * edge2.z,
      z: edge1.x * edge2.y - edge1.y * edge2.x,
    };

    const length = Math.sqrt(normal.x ** 2 + normal.y ** 2 + normal.z ** 2);
    if (length > 0) {
      normal.x /= length;
      normal.y /= length;
      normal.z /= length;
    }

    return normal;
  }

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
    setRotation({ x: 30, y: 45 });
    setZoom(1);
  };

  return (
    <div className="w-full h-full relative" data-testid="mesh-viewer">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Processing Overlay */}
      {isProcessing && (
        <div className="absolute inset-0 bg-[var(--obsidian)]/90 flex flex-col items-center justify-center">
          <div className="w-16 h-16 border-3 border-[var(--surface-variant)] border-t-[var(--copper-base)] rounded-full animate-spin mb-4" />
          <p className="text-sm text-[var(--on-surface)]">Generating mesh...</p>
        </div>
      )}

      {/* Empty State */}
      {!meshData && !isProcessing && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <Layers className="w-16 h-16 text-[var(--surface-variant)] mb-4" />
          <p className="text-lg font-medium text-[var(--on-surface)] mb-2">
            Mesh Preview
          </p>
          <p className="text-sm text-[var(--outline)]">
            Select Triangle renderer and process images
          </p>
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 p-2 bg-[var(--surface)]/85 backdrop-blur-md border border-[var(--surface-variant)] rounded-xl">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="w-10 h-10 flex items-center justify-center rounded-lg border border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)] transition-colors"
                onClick={handleReset}
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Reset View</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="w-10 h-10 flex items-center justify-center rounded-lg border border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)] transition-colors"
                onClick={() => setZoom((z) => Math.min(3, z * 1.2))}
              >
                <ZoomIn className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Zoom In</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="w-10 h-10 flex items-center justify-center rounded-lg border border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)] transition-colors"
                onClick={() => setZoom((z) => Math.max(0.5, z * 0.8))}
              >
                <ZoomOut className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Zoom Out</TooltipContent>
          </Tooltip>

          <div className="w-px h-6 bg-[var(--surface-variant)] self-center mx-1" />

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`w-10 h-10 flex items-center justify-center rounded-lg border transition-colors ${
                  showFaces
                    ? "bg-[var(--crimson)] border-[var(--crimson)] text-white"
                    : "border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)]"
                }`}
                onClick={() => setShowFaces(!showFaces)}
              >
                <Box className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Toggle Faces</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`w-10 h-10 flex items-center justify-center rounded-lg border transition-colors ${
                  showWireframe
                    ? "bg-[var(--copper-base)] border-[var(--copper-base)] text-white"
                    : "border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)]"
                }`}
                onClick={() => setShowWireframe(!showWireframe)}
              >
                <Layers className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Toggle Wireframe</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className={`w-10 h-10 flex items-center justify-center rounded-lg border transition-colors ${
                  showGrid
                    ? "bg-[var(--surface-variant)] border-[var(--outline)]"
                    : "border-[var(--surface-variant)] text-[var(--on-surface)] hover:bg-[var(--surface-variant)]"
                }`}
                onClick={() => setShowGrid(!showGrid)}
              >
                <Grid3X3 className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Toggle Grid</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Rotation Info */}
      {meshData && (
        <div className="absolute top-4 left-4 bg-[var(--surface)]/80 backdrop-blur-sm px-3 py-2 rounded-lg">
          <p className="text-[0.65rem] text-[var(--outline)] font-mono">
            X: {Math.round(rotation.x)}° Y: {Math.round(rotation.y)}° Z: {(zoom * 100).toFixed(0)}%
          </p>
        </div>
      )}
    </div>
  );
}
