import React from "react";
import { Settings, Info, Triangle, Circle } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function SettingsPanel({ settings, onSettingsChange, disabled }) {
  const updateSetting = (key, value) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <div className="control-panel" data-testid="settings-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <Settings className="w-4 h-4" />
          Settings
        </h2>
      </div>

      <div className="panel-content">
        {/* Renderer Type */}
        <div className="settings-section">
          <div className="flex items-center gap-2 mb-2">
            <Label className="settings-label mb-0">Renderer</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-3 h-3 text-[var(--outline)]" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p><strong>Gaussian:</strong> INRIA 3DGS - smooth, view-dependent colors</p>
                  <p className="mt-1"><strong>Triangle:</strong> 3DV 2026 - sharper edges, game engine compatible</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => updateSetting("renderer", "gaussian")}
              disabled={disabled}
              className={`flex items-center justify-center gap-2 p-3 rounded-lg border transition-all ${
                settings.renderer === "gaussian"
                  ? "border-[var(--crimson)] bg-[var(--crimson)]/10 text-[var(--crimson)]"
                  : "border-[var(--surface-variant)] text-[var(--outline)] hover:border-[var(--copper-base)]"
              }`}
              data-testid="renderer-gaussian"
            >
              <Circle className="w-4 h-4" />
              <span className="text-xs font-medium">Gaussian</span>
            </button>
            <button
              onClick={() => updateSetting("renderer", "triangle")}
              disabled={disabled}
              className={`flex items-center justify-center gap-2 p-3 rounded-lg border transition-all ${
                settings.renderer === "triangle"
                  ? "border-[var(--copper-base)] bg-[var(--copper-base)]/10 text-[var(--copper-base)]"
                  : "border-[var(--surface-variant)] text-[var(--outline)] hover:border-[var(--copper-base)]"
              }`}
              data-testid="renderer-triangle"
            >
              <Triangle className="w-4 h-4" />
              <span className="text-xs font-medium">Triangle</span>
            </button>
          </div>
          <p className="text-[0.6rem] text-[var(--outline)] mt-2">
            {settings.renderer === "triangle" 
              ? "Triangle Splatting (3DV 2026) - sharper, 2400+ FPS, mesh export"
              : "3D Gaussian Splatting (SIGGRAPH 2023) - smooth, view-dependent"}
          </p>
        </div>

        {/* Quality Preset */}
        <div className="settings-section">
          <div className="flex items-center gap-2 mb-2">
            <Label className="settings-label mb-0">Quality Preset</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-3 h-3 text-[var(--outline)]" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Higher quality = longer processing time</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Select
            value={settings.quality}
            onValueChange={(value) => updateSetting("quality", value)}
            disabled={disabled}
          >
            <SelectTrigger
              className="w-full bg-[var(--secondary)] border-[var(--surface-variant)]"
              data-testid="quality-select"
            >
              <SelectValue placeholder="Select quality" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="low">Low (Fast)</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="ultra">Ultra (Slow)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Resolution */}
        <div className="settings-section">
          <div className="flex items-center justify-between mb-2">
            <Label className="settings-label mb-0">Resolution</Label>
            <span className="text-xs text-[var(--copper-base)] font-mono">
              {settings.resolution}px
            </span>
          </div>
          <Slider
            value={[settings.resolution]}
            onValueChange={([value]) => updateSetting("resolution", value)}
            min={512}
            max={2048}
            step={256}
            disabled={disabled}
            className="py-2"
            data-testid="resolution-slider"
          />
          <div className="flex justify-between text-[0.65rem] text-[var(--outline)] mt-1">
            <span>512</span>
            <span>2048</span>
          </div>
        </div>

        {/* Iterations */}
        <div className="settings-section">
          <div className="flex items-center justify-between mb-2">
            <Label className="settings-label mb-0">Training Iterations</Label>
            <span className="text-xs text-[var(--copper-base)] font-mono">
              {(settings.iterations / 1000).toFixed(0)}K
            </span>
          </div>
          <Slider
            value={[settings.iterations]}
            onValueChange={([value]) => updateSetting("iterations", value)}
            min={7000}
            max={50000}
            step={1000}
            disabled={disabled}
            className="py-2"
            data-testid="iterations-slider"
          />
          <div className="flex justify-between text-[0.65rem] text-[var(--outline)] mt-1">
            <span>7K</span>
            <span>50K</span>
          </div>
        </div>

        {/* SH Degree - Only for Gaussian */}
        {settings.renderer === "gaussian" && (
          <div className="settings-section">
            <div className="flex items-center gap-2 mb-2">
              <Label className="settings-label mb-0">SH Degree</Label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="w-3 h-3 text-[var(--outline)]" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Spherical harmonics degree for view-dependent effects</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Select
              value={String(settings.sh_degree)}
              onValueChange={(value) => updateSetting("sh_degree", parseInt(value))}
              disabled={disabled}
            >
              <SelectTrigger
                className="w-full bg-[var(--secondary)] border-[var(--surface-variant)]"
                data-testid="sh-degree-select"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">0 (Diffuse only)</SelectItem>
                <SelectItem value="1">1 (Basic)</SelectItem>
                <SelectItem value="2">2 (Standard)</SelectItem>
                <SelectItem value="3">3 (High detail)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
    </div>
  );
}
