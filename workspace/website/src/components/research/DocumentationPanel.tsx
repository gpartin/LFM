/**
 * DocumentationPanel component for research experiments.
 * Displays links to related documentation, config files, results, and discovery docs.
 * 
 * Enables full scientific traceability: every experiment links to its configuration,
 * validation results, and theoretical documentation.
 */

'use client';

import React from 'react';
import Link from 'next/link';
import { ExperimentDefinition } from '@/data/experiments';

interface DocumentationPanelProps {
  experiment: ExperimentDefinition;
}

export default function DocumentationPanel({ experiment }: DocumentationPanelProps) {
  const { links } = experiment;
  
  if (!links) {
    return null;
  }
  
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">📚 Documentation & Evidence</h3>
      
      <div className="space-y-4">
        {/* Configuration file (test harness) */}
        {links.testHarnessConfig && (
          <div>
            <h4 className="text-sm font-medium text-white/70 mb-2">Configuration</h4>
            <div className="space-y-1">
              <DocumentationLink
                href={links.testHarnessConfig}
                label="Test Configuration"
                icon="⚙️"
              />
            </div>
          </div>
        )}
        
        {/* Results directory */}
        {links.results && (
          <div>
            <h4 className="text-sm font-medium text-white/70 mb-2">Results</h4>
            <DocumentationLink
              href={links.results}
              label="Validation Results"
              icon="📊"
            />
          </div>
        )}
        
        {/* Discovery documentation */}
        {links.discovery && (
          <div>
            <h4 className="text-sm font-medium text-white/70 mb-2">Discovery</h4>
            {Array.isArray(links.discovery) ? (
              links.discovery.map((discoveryPath, i) => (
                <DocumentationLink
                  key={i}
                  href={discoveryPath}
                  label={`Discovery Doc ${i + 1}`}
                  icon="🔬"
                />
              ))
            ) : (
              <DocumentationLink
                href={links.discovery}
                label="Discovery Documentation"
                icon="🔬"
              />
            )}
          </div>
        )}
        
        {/* Additional documentation */}
        {links.documentation && (
          <div>
            <h4 className="text-sm font-medium text-white/70 mb-2">Documentation</h4>
            <DocumentationLink
              href={links.documentation}
              label="Technical Documentation"
              icon="📄"
              external={links.documentation.startsWith('http://') || links.documentation.startsWith('https://')}
            />
          </div>
        )}
      </div>
      
      {/* Reproducibility statement */}
      <div className="mt-6 pt-4 border-t border-white/10">
        <p className="text-xs text-white/50 leading-relaxed">
          All linked resources are versioned and traceable. Configuration files define exact
          simulation parameters, results contain raw data and diagnostics, and discovery docs
          provide theoretical context. This ensures full reproducibility and scientific transparency.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Helper Components
// ============================================================================

interface DocumentationLinkProps {
  href: string;
  label: string;
  icon: string;
  external?: boolean;
}

function DocumentationLink({ href, label, icon, external = false }: DocumentationLinkProps) {
  // Check if href is external URL or local path
  const isExternal = external || href.startsWith('http://') || href.startsWith('https://');
  
  if (isExternal) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
      >
        <span>{icon}</span>
        <span>{label}</span>
        <span className="text-white/30">↗</span>
      </a>
    );
  }
  
  // For local paths, link to a file viewer or download
  // For now, display as non-interactive (file serving not yet implemented)
  return (
    <div className="flex items-center gap-2 text-sm text-white/50">
      <span>{icon}</span>
      <span>{label}</span>
      <code className="ml-auto text-xs text-white/30 font-mono">{href}</code>
    </div>
  );
}
