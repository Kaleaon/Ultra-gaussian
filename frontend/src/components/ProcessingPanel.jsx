import React from "react";
import { Activity, Play, AlertCircle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

export default function ProcessingPanel({
  processingStatus,
  imageCount,
  onStartProcessing,
  isLoading,
  isProcessing,
}) {
  const progress = processingStatus?.progress || 0;
  const currentStep = processingStatus?.current_step || "Ready";
  const status = processingStatus?.status || "idle";

  const getStatusDisplay = () => {
    switch (status) {
      case "queued":
        return { text: "Queued", color: "text-[var(--outline)]", icon: Activity };
      case "preprocessing":
        return { text: "Preprocessing", color: "text-[var(--copper-base)]", icon: Activity };
      case "training":
        return { text: "Training", color: "text-[var(--copper-highlight)]", icon: Activity };
      case "postprocessing":
        return { text: "Finalizing", color: "text-[var(--copper-shimmer)]", icon: Activity };
      case "completed":
        return { text: "Completed", color: "text-green-500", icon: CheckCircle2 };
      case "failed":
        return { text: "Failed", color: "text-[var(--crimson)]", icon: AlertCircle };
      default:
        return { text: "Ready", color: "text-[var(--outline)]", icon: Activity };
    }
  };

  const statusDisplay = getStatusDisplay();
  const StatusIcon = statusDisplay.icon;
  const canStart = imageCount >= 3 && !isProcessing && status !== "completed";

  return (
    <div className="control-panel" data-testid="processing-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <Activity className="w-4 h-4" />
          Processing
        </h2>
        <div className={`flex items-center gap-2 text-xs ${statusDisplay.color}`}>
          <StatusIcon className="w-3 h-3" />
          {statusDisplay.text}
        </div>
      </div>

      <div className="panel-content">
        {/* Status Widget */}
        <div className="status-widget">
          <div
            className={`status-indicator ${
              isProcessing ? "processing" : status === "completed" ? "active" : "idle"
            }`}
          />
          <div className="flex-1">
            <div className="status-label">Current Step</div>
            <div className="status-value">{currentStep}</div>
          </div>
        </div>

        {/* Progress Bar */}
        {(isProcessing || status === "completed") && (
          <div className="progress-container" data-testid="processing-progress-bar">
            <div className="progress-bar">
              <div
                className={`progress-fill ${isProcessing ? "shimmer" : ""}`}
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="progress-label">
              <span className="progress-step">{currentStep}</span>
              <span className="progress-percent">{Math.round(progress)}%</span>
            </div>
          </div>
        )}

        {/* Info/Warning */}
        {imageCount < 3 && (
          <div className="flex items-start gap-2 p-3 bg-[var(--secondary)] rounded-lg mb-4">
            <AlertCircle className="w-4 h-4 text-[var(--crimson)] mt-0.5 flex-shrink-0" />
            <p className="text-xs text-[var(--outline)]">
              Upload at least 3 images to start reconstruction. 10-50 images recommended for best results.
            </p>
          </div>
        )}

        {/* Start Button */}
        <Button
          onClick={onStartProcessing}
          disabled={!canStart || isLoading}
          className="w-full bg-[var(--crimson)] hover:bg-[var(--crimson-dark)] text-white border-0 h-12 text-sm font-medium tracking-wide"
          data-testid="start-processing-btn"
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : isProcessing ? (
            <>
              <Activity className="w-4 h-4 mr-2 animate-pulse" />
              Processing...
            </>
          ) : status === "completed" ? (
            <>
              <CheckCircle2 className="w-4 h-4 mr-2" />
              Complete
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Start Processing
            </>
          )}
        </Button>

        {/* NPU Notice */}
        {!isProcessing && status !== "completed" && (
          <p className="text-[0.65rem] text-[var(--outline)] text-center mt-3">
            Processing uses WebNN for NPU/GPU acceleration when available
          </p>
        )}
      </div>
    </div>
  );
}
