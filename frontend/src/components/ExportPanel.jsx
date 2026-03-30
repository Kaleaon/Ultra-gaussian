import React from "react";
import { Download, FileBox, FileCode, File, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function ExportPanel({ onExport, disabled, renderer = "gaussian" }) {
  const gaussianFormats = [
    {
      id: "ply",
      label: "PLY",
      description: "Point cloud format, ideal for 3D software",
      icon: FileBox,
    },
    {
      id: "gltf",
      label: "GLTF",
      description: "Web-ready 3D format for browsers",
      icon: FileCode,
    },
    {
      id: "obj",
      label: "OBJ",
      description: "Universal 3D mesh format",
      icon: File,
    },
  ];

  const triangleFormats = [
    {
      id: "off",
      label: "OFF",
      description: "Object File Format - game engine ready",
      icon: Layers,
    },
    {
      id: "obj",
      label: "OBJ",
      description: "Universal mesh with materials",
      icon: File,
    },
    {
      id: "gltf",
      label: "GLTF",
      description: "Web-ready 3D format",
      icon: FileCode,
    },
  ];

  const formats = renderer === "triangle" ? triangleFormats : gaussianFormats;

  return (
    <div className="control-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <Download className="w-4 h-4" />
          Export Model
        </h2>
      </div>

      <div className="panel-content">
        {disabled && (
          <p className="text-xs text-[var(--outline)] mb-4">
            Complete processing to enable export options
          </p>
        )}

        <div className="export-grid">
          <TooltipProvider>
            {formats.map((format) => {
              const Icon = format.icon;
              return (
                <Tooltip key={format.id}>
                  <TooltipTrigger asChild>
                    <button
                      className="export-btn"
                      onClick={() => onExport(format.id)}
                      disabled={disabled}
                      data-testid={`export-btn-${format.id}`}
                    >
                      <Icon className="w-4 h-4 mx-auto mb-1" />
                      {format.label}
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{format.description}</p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </TooltipProvider>
        </div>

        {!disabled && (
          <p className="text-[0.65rem] text-[var(--outline)] text-center mt-4">
            {renderer === "triangle" 
              ? "OFF format is game engine compatible"
              : "Exported models retain full Gaussian splat data"
            }
          </p>
        )}
      </div>
    </div>
  );
}
