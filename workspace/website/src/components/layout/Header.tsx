'use client';

import Link from 'next/link';

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-space-dark/95 backdrop-blur-sm border-b border-space-border">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-accent-chi to-accent-particle rounded-lg flex items-center justify-center">
              <span className="text-2xl">⚛️</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-accent-chi">Emergent Physics Lab</h1>
              <p className="text-xs text-text-secondary">Where forces emerge from the lattice</p>
            </div>
          </Link>

          {/* Research Links */}
          <nav className="hidden md:flex items-center space-x-4">
            <ExternalLink
              href="https://osf.io/6agn8"
              label="OSF Project"
              badge="10.17605/OSF.IO/6AGN8"
              icon="🔬"
            />
            <ExternalLink
              href="https://zenodo.org/records/17536484"
              label="Zenodo"
              badge="10.5281/zenodo.17536484"
              icon="📚"
            />
            <ExternalLink
              href="https://github.com/gpartin/LFM"
              label="GitHub"
              badge="91.4% Tests Pass"
              icon="💻"
            />
            <Link
              href="/about"
              className="px-4 py-2 text-text-secondary hover:text-accent-chi transition-colors"
            >
              About
            </Link>
          </nav>

          {/* Mobile menu button */}
          <button className="md:hidden p-2 text-text-secondary hover:text-accent-chi">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}

function ExternalLink({ href, label, badge, icon }: { href: string; label: string; badge: string; icon: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex flex-col items-start px-4 py-2 rounded-lg hover:bg-space-panel transition-all"
    >
      <div className="flex items-center space-x-2">
        <span>{icon}</span>
        <span className="text-sm font-semibold text-text-primary group-hover:text-accent-chi transition-colors">
          {label}
        </span>
      </div>
      <span className="text-xs text-text-muted font-mono">{badge}</span>
    </a>
  );
}
