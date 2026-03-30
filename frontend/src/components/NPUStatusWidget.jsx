import React from "react";
import { Cpu, Activity, Zap, AlertTriangle } from "lucide-react";

export default function NPUStatusWidget({ capabilities, isProcessing }) {
  const hasNPU = capabilities?.npuAvailable;
  const hasGPU = capabilities?.webgpu;
  const hasWebNN = capabilities?.webnn;
  const backend = capabilities?.backend || "cpu";

  const getBackendDisplay = () => {
    if (hasNPU) return { label: "NPU", color: "text-green-400", icon: Zap };
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
            isProcessing ? "processing" : hasNPU || hasGPU ? "active" : "idle"
          }`}
        />
      </div>

      <div className="npu-metrics">
        <div className="npu-metric">
          <div className={`npu-metric-value ${backendDisplay.color}`}>
            <BackendIcon className="w-5 h-5 inline mr-1" />
            {backendDisplay.label}
          </div>
          <div className="npu-metric-label">Active Backend</div>
        </div>

        <div className="npu-metric">
          <div
            className={`npu-metric-value ${
              hasWebNN ? "text-green-400" : "text-[var(--outline)]"
            }`}
          >
            {hasWebNN ? "YES" : "NO"}
          </div>
          <div className="npu-metric-label">WebNN API</div>
        </div>
      </div>

      {/* Capability Badges */}
      <div className="flex gap-2 mt-4">
        <div
          className={`text-[0.625rem] px-2 py-1 rounded-full ${
            hasGPU
              ? "bg-green-500/10 text-green-400 border border-green-500/20"
              : "bg-[var(--surface-variant)] text-[var(--outline)]"
          }`}
        >
          WebGPU
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
          NPU
        </div>
      </div>

      {/* Warning if no acceleration */}
      {!hasGPU && !hasNPU && (
        <div className="flex items-start gap-2 mt-4 p-2 bg-[var(--surface-variant)] rounded text-[0.65rem] text-[var(--outline)]">
          <AlertTriangle className="w-3 h-3 text-yellow-500 mt-0.5 flex-shrink-0" />
          <span>
            No hardware acceleration detected. Processing will use CPU (slower).
          </span>
        </div>
      )}
    </div>
  );
}
