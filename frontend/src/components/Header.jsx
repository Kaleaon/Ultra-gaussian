import React from "react";
import { Box, Plus, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function Header({ projectName, onNewProject }) {
  return (
    <header className="app-header" data-testid="app-header">
      <div className="logo-section">
        <div className="logo-mark">
          <Box className="w-5 h-5 text-white" />
        </div>
        <h1 className="logo-text">
          Instant<span>3D</span>
        </h1>
      </div>

      <div className="flex items-center gap-4">
        <span className="text-sm text-[var(--outline)] hidden sm:block">
          {projectName}
        </span>
        
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={onNewProject}
                className="border-[var(--surface-variant)] bg-transparent hover:bg-[var(--surface-variant)] hover:border-[var(--copper-base)]"
                data-testid="new-project-btn"
              >
                <Plus className="w-4 h-4 mr-2" />
                New Project
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Create a new 3D reconstruction project</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-9 h-9 flex items-center justify-center rounded-lg border border-[var(--surface-variant)] text-[var(--outline)] hover:text-[var(--on-surface)] hover:border-[var(--copper-base)] transition-all"
                data-testid="github-link"
              >
                <Github className="w-4 h-4" />
              </a>
            </TooltipTrigger>
            <TooltipContent>
              <p>View on GitHub</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </header>
  );
}
