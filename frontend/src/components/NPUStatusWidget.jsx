import React, { useEffect, useState } from "react";
import { Cpu, Activity, Zap, AlertTriangle, Smartphone, ChevronDown, ChevronUp } from "lucide-react";
import { detectDeviceCapabilities } from "@/utils/webnnClassifier";

export default function NPUStatusWidget({ capabilities, isProcessing }) {
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [isDetecting, setIsDetecting] = useState(true);
  const [showGuide, setShowGuide] = useState(false);

  useEffect(() => {
    async function detect() {
      try {
        const info = await detectDeviceCapabilities();
        setDeviceInfo(info);
      } catch (e) {
        console.error('Device detection failed:', e);
      } finally {
        setIsDetecting(false);
      }
    }
    detect();
  }, []);

  // Merge browser capabilities with device info
  const hasNPU = deviceInfo?.npuAvailable || capabilities?.npuAvailable;
  const hasGPU = deviceInfo?.gpuAvailable || deviceInfo?.webgpu || capabilities?.webgpu;
  const hasWebNN = deviceInfo?.webnn || capabilities?.webnn;
  const isPixel = deviceInfo?.isPixel;
  const tensorVersion = deviceInfo?.tensorVersion;

  const getBackendDisplay = () => {
    if (hasNPU) return { label: isPixel ? "TPU" : "NPU", color: "text-green-400", icon: Zap };
    if (hasGPU) return { label: "GPU", color: "text-[var(--copper-base)]", icon: Activity };
    return { label: "CPU", color: "text-[var(--outline)]", icon: Cpu };
  };

  const backendDisplay = getBackendDisplay();
  const BackendIcon = backendDisplay.icon;

  return (
    <div
      className={`npu-card ${isProcessing || hasNPU ? "active" : ""}`}
      data-testid="npu-status-card"
    >
      <div className="npu-header">
        <div className="npu-title">
          <Cpu className="w-4 h-4" />
          Device Acceleration
        </div>
        <div
          className={`status-indicator ${
            isDetecting ? "processing" : isProcessing ? "processing" : hasNPU || hasGPU ? "active" : "idle"
          }`}
        />
      </div>

      {/* Pixel Device Badge */}
      {isPixel && (
        <div className="flex items-center gap-2 mb-3 p-2 bg-[var(--surface-variant)] rounded-lg">
          <Smartphone className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-[var(--on-surface)]">
            Google Pixel {tensorVersion && `(Tensor ${tensorVersion})`}
          </span>
        </div>
      )}

      <div className="npu-metrics">
        <div className="npu-metric">
          <div className={`npu-metric-value ${backendDisplay.color}`}>
            <BackendIcon className="w-5 h-5 inline mr-1" />
            {isDetecting ? "..." : backendDisplay.label}
          </div>
          <div className="npu-metric-label">Active Backend</div>
        </div>

        <div className="npu-metric">
          <div
            className={`npu-metric-value ${
              hasWebNN ? "text-green-400" : "text-[var(--outline)]"
            }`}
          >
            {isDetecting ? "..." : hasWebNN ? "YES" : "NO"}
          </div>
          <div className="npu-metric-label">WebNN API</div>
        </div>
      </div>

      {/* Capability Badges */}
      <div className="flex flex-wrap gap-2 mt-4">
        <div
          className={`text-[0.625rem] px-2 py-1 rounded-full ${
            hasGPU
              ? "bg-green-500/10 text-green-400 border border-green-500/20"
              : "bg-[var(--surface-variant)] text-[var(--outline)]"
          }`}
        >
          {deviceInfo?.webgpu ? "WebGPU" : "GPU"}
        </div>
        <div
          className={`text-[0.625rem] px-2 py-1 rounded-full ${
            hasWebNN
              ? "bg-green-500/10 text-green-400 border border-green-500/20"
              : "bg-[var(--surface-variant)] text-[var(--outline)]"
          }`}
        >
          WebNN
        </div>
        <div
          className={`text-[0.625rem] px-2 py-1 rounded-full ${
            hasNPU
              ? "bg-[var(--copper-base)]/10 text-[var(--copper-highlight)] border border-[var(--copper-base)]/20"
              : "bg-[var(--surface-variant)] text-[var(--outline)]"
          }`}
        >
          {isPixel ? "Tensor TPU" : "NPU"}
        </div>
      </div>

      {/* Instructions for enabling WebNN on Pixel */}
      {isPixel && !hasWebNN && (
        <div className="flex items-start gap-2 mt-4 p-2 bg-blue-500/10 rounded text-[0.65rem] text-blue-400">
          <Smartphone className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>
            Enable TPU: Open <code className="bg-[var(--surface-variant)] px-1 rounded">chrome://flags</code> → Search "WebNN" → Enable → Restart Chrome
          </span>
        </div>
      )}

      {/* Warning if no acceleration */}
      {!isDetecting && !hasGPU && !hasNPU && (
        <div className="flex items-start gap-2 mt-4 p-2 bg-[var(--surface-variant)] rounded text-[0.65rem] text-[var(--outline)]">
          <AlertTriangle className="w-3 h-3 text-yellow-500 mt-0.5 flex-shrink-0" />
          <span>
            No hardware acceleration detected. Processing will use CPU (slower).
          </span>
        </div>
      )}

      {/* Pixel TPU Setup Guide Toggle */}
      <button
        onClick={() => setShowGuide(!showGuide)}
        className="w-full flex items-center justify-between mt-4 p-2 bg-[var(--surface-variant)]/50 hover:bg-[var(--surface-variant)] rounded text-[0.65rem] text-[var(--outline)] transition-colors"
        data-testid="tpu-guide-toggle"
      >
        <span>Pixel TPU Setup Guide</span>
        {showGuide ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {showGuide && (
        <div className="mt-2 p-3 bg-[var(--secondary)] rounded-lg text-[0.6rem] text-[var(--outline)] space-y-2" data-testid="tpu-guide-content">
          <p className="text-[var(--copper-base)] font-semibold text-[0.65rem]">Google Pixel TPU Testing</p>
          <div className="space-y-1.5">
            <p><span className="text-[var(--on-surface)]">1.</span> Open Chrome on your Pixel device (6 or newer)</p>
            <p><span className="text-[var(--on-surface)]">2.</span> Navigate to <code className="bg-[var(--surface-variant)] px-1 rounded">chrome://flags</code></p>
            <p><span className="text-[var(--on-surface)]">3.</span> Search for <code className="bg-[var(--surface-variant)] px-1 rounded">WebNN API</code></p>
            <p><span className="text-[var(--on-surface)]">4.</span> Set to <span className="text-green-400">Enabled</span> and restart Chrome</p>
            <p><span className="text-[var(--on-surface)]">5.</span> Visit this app &mdash; the widget above will show <span className="text-green-400">TPU</span></p>
          </div>
          <hr className="border-[var(--surface-variant)]" />
          <p className="text-[var(--outline)]">
            The Tensor chip TPU is accessed via Android NNAPI &rarr; LiteRT &rarr; WebNN.
            Image classification runs MobileNetV2 (ONNX) locally on your device &mdash; no server round-trip.
            People in frames are automatically masked out (inpainted) instead of discarded.
          </p>
          <div className="mt-1">
            <p className="text-[var(--copper-base)]">Supported Pixels:</p>
            <p>Pixel 6/6a/7/7a/8/8a/9 (Tensor G1&ndash;G5)</p>
          </div>
        </div>
      )}
    </div>
  );
}
