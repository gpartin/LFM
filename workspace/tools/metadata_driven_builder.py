#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Metadata-Driven Upload Builder

This is the new metadata-driven upload builder that uses the comprehensive
schema to build deterministic, validated upload packages for both Zenodo and OSF.

KEY FEATURES:
1. Schema-driven: Uses upload_metadata_schema.py to define all files
2. Platform-aware: Builds common + platform-specific files  
3. Validated: Automatically validates output against schema
4. Deterministic: Same inputs always produce same outputs
5. Traceable: Every file traced back to its source

DIRECTORY STRUCTURE:
upload/
├── [19 COMMON FILES] - Shared by both platforms
├── zenodo/
│   ├── zenodo_metadata.json
│   └── ZENODO_UPLOAD.zip
└── osf/
    ├── osf_project.json
    └── OSF_UPLOAD.zip

USAGE:
    python tools/metadata_driven_builder.py
    python tools/metadata_driven_builder.py --validate-only
    python tools/metadata_driven_builder.py --platform zenodo
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from upload_metadata_schema import UploadMetadataSchema, FileMetadata, FileSource
from text_template_engine import TextTemplateEngine
from upload_validator import UploadValidator

class MetadataDrivenBuilder:
    """Builds upload directory using metadata schema"""
    
    def __init__(self, project_root: Path, upload_dir: Path, enforce_review: bool = False):
        self.project_root = project_root
        self.upload_dir = upload_dir
        self.schema = UploadMetadataSchema()
        self.build_log = []
        self.enforce_review = enforce_review
        self.template_engine = TextTemplateEngine(self.project_root)
        
    def build_complete_package(self, platforms: Optional[List[str]] = None) -> bool:
        """Build complete upload package"""
        platforms = platforms or ['zenodo', 'osf']
        
        print("🏗️  Metadata-Driven Upload Builder")
        print("=" * 50)
        print(f"Schema Version: {self.schema.schema_version}")
        print(f"Target Platforms: {', '.join(platforms)}")
        print()
        
        # Step 1: Clean slate
        self._clean_upload_directory()
        
        # Step 2: Build common files
        success = self._build_common_files()
        if not success:
            print("❌ Failed to build common files")
            return False
        
        # Step 3: Build platform-specific files
        for platform in platforms:
            success = self._build_platform_files(platform)
            if not success:
                print(f"❌ Failed to build {platform} files")
                return False
        
        # Step 4: Validate output
        success = self._validate_output()
        if not success:
            print("❌ Validation failed")
            return False
        
        # Step 5: Generate final bundles
        self._generate_final_bundles(platforms)
        
        print("\n" + "=" * 50)
        print("✅ METADATA-DRIVEN BUILD COMPLETE!")
        print(f"📁 Location: {self.upload_dir}")
        print(f"📋 Files Generated: {self._count_generated_files()}")
        print(f"🎯 Ready for upload to: {', '.join(platforms)}")
        
        return True
    
    def _clean_upload_directory(self):
        """Clean slate approach - remove and recreate upload directory"""
        if self.upload_dir.exists():
            shutil.rmtree(self.upload_dir)
        
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create platform subdirectories
        (self.upload_dir / 'zenodo').mkdir(exist_ok=True)
        (self.upload_dir / 'osf').mkdir(exist_ok=True)
        (self.upload_dir / 'plots').mkdir(exist_ok=True)
        
        print("🗑️  Cleaned upload directory")
        self.build_log.append("✓ Cleaned upload directory")
    
    def _build_common_files(self) -> bool:
        """Build all 'both' files (shared by both platforms) into zenodo/ and osf/ directories"""
        print("📋 Building shared files for both platforms...")
        
        both_files = self.schema.get_both_platform_files()
        build_order = self._get_build_order(both_files)
        
        for filename in build_order:
            metadata = both_files[filename]
            # Build file into zenodo/ directory
            zenodo_path = f"zenodo/{filename}"
            success_zenodo = self._build_single_file(zenodo_path, metadata, filename)
            if not success_zenodo and metadata.required:
                print(f"❌ Failed to build required file in zenodo/: {filename}")
                return False
            
            # Build file into osf/ directory
            osf_path = f"osf/{filename}"
            success_osf = self._build_single_file(osf_path, metadata, filename)
            if not success_osf and metadata.required:
                print(f"❌ Failed to build required file in osf/: {filename}")
                return False
        
        return True
    
    def _build_platform_files(self, platform: str) -> bool:
        """Build platform-specific files"""
        print(f"🔬 Building {platform} files...")
        
        if platform == 'zenodo':
            platform_files = self.schema.get_zenodo_files()
        elif platform == 'osf':
            platform_files = self.schema.get_osf_files()
        else:
            print(f"❌ Unknown platform: {platform}")
            return False
        
        for filename, metadata in platform_files.items():
            success = self._build_single_file(filename, metadata)
            if not success and metadata.required:
                print(f"❌ Failed to build required {platform} file: {filename}")
                return False
        
        return True
    
    def _build_single_file(self, dest_filename: str, metadata: FileMetadata, source_filename: str = None) -> bool:
        """Build a single file according to its metadata
        
        Args:
            dest_filename: Full destination path relative to upload_dir (e.g., 'zenodo/README.md')
            metadata: File metadata defining how to build it
            source_filename: Original filename without platform prefix (e.g., 'README.md'), used for display
        """
        if source_filename is None:
            source_filename = dest_filename
            
        source = metadata.source
        dest_path = self.upload_dir / dest_filename
        
        try:
            if source.source_type == 'copy':
                return self._copy_file(source, dest_path, source_filename)
            elif source.source_type == 'convert':
                return self._convert_file(source, dest_path, source_filename)
            elif source.source_type == 'generate':
                return self._generate_file(source, dest_path, source_filename)
            else:
                print(f"⚠️  Unknown source type for {source_filename}: {source.source_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error building {source_filename}: {e}")
            return False
    
    def _copy_file(self, source: FileSource, dest_path: Path, filename: str) -> bool:
        """Copy file from source location"""
        if not source.source_path:
            print(f"❌ No source path for copy operation: {filename}")
            return False
        
        # Handle glob patterns in source path
        source_pattern = source.source_path
        if '*' in source_pattern:
            # Find matching files
            matches = list(self.project_root.glob(source_pattern))
            if not matches:
                print(f"⚠️  No files found matching pattern: {source_pattern}")
                return False
            source_file = matches[0]  # Use first match
        else:
            source_file = self.project_root / source_pattern
        
        if not source_file.exists():
            print(f"⚠️  Source file not found: {source_file}")
            return False
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_file, dest_path)
        print(f"✓ Copied {filename}")
        self.build_log.append(f"✓ Copied {filename} from {source_file}")
        return True
    
    def _convert_file(self, source: FileSource, dest_path: Path, filename: str) -> bool:
        """Convert text file to markdown"""
        if not source.source_path:
            print(f"❌ No source path for convert operation: {filename}")
            return False
        
        source_file = self.project_root / source.source_path
        if not source_file.exists():
            print(f"⚠️  Source file not found: {source_file}")
            return False
        
        # Read source content
        try:
            raw = source_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"❌ Cannot read source file {source_file}: {e}")
            return False
        
        # Preprocess templates (includes and dynamic tokens)
        content = self.template_engine.render(raw)

        # Convert to markdown with header
        title = filename.replace('.md', '').replace('_', ' ')
        markdown_content = self._generate_markdown_header(title) + '\n\n' + content
        
        # Write converted content
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(markdown_content, encoding='utf-8')
        
        print(f"✓ Converted {filename}")
        self.build_log.append(f"✓ Converted {filename} from {source_file}")
        return True
    
    def _generate_file(self, source: FileSource, dest_path: Path, filename: str) -> bool:
        """Generate file using specified function"""
        if not source.generator_function:
            print(f"❌ No generator function for: {filename}")
            return False
        
        generator_map = {
            'generate_comprehensive_results': self._generate_comprehensive_results,
            'generate_electromagnetic_achievements': self._generate_electromagnetic_achievements,
            'generate_evidence_review': self._generate_evidence_review,
            'generate_comprehensive_pdf': self._generate_comprehensive_pdf,
            'generate_pdf_from_text': self._generate_pdf_from_text,
            'generate_plots_overview': self._generate_plots_overview,
            'generate_discoveries_overview': self._generate_discoveries_overview,
            'generate_manifest': self._generate_manifest,
            'generate_zenodo_metadata': self._generate_zenodo_metadata,
            'generate_osf_metadata': self._generate_osf_metadata,
            'create_zenodo_bundle': self._create_zenodo_bundle,
            'create_osf_bundle': self._create_osf_bundle
        }
        
        generator = generator_map.get(source.generator_function)
        if not generator:
            print(f"❌ Unknown generator function: {source.generator_function}")
            return False
        
        try:
            success = generator(dest_path, filename)
            if success:
                print(f"✓ Generated {filename}")
                self.build_log.append(f"✓ Generated {filename}")
            return success
        except Exception as e:
            print(f"❌ Generator failed for {filename}: {e}")
            return False
    
    def _generate_markdown_header(self, title: str) -> str:
        """Generate standard markdown header"""
        return f"""---
title: "{title}"
author: "Greg D. Partin"
institution: "LFM Research, Los Angeles CA USA"
license: "CC BY-NC-ND 4.0"
contact: "latticefieldmediumresearch@gmail.com"
orcid: "https://orcid.org/0009-0004-0327-6528"
doi: "10.5281/zenodo.17510124"
generated: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
---

# {title}"""
    
    def _generate_comprehensive_results(self, dest_path: Path, filename: str) -> bool:
        """Generate comprehensive results document from results/* directories"""
        results_dir = self.project_root / 'results'
        config_path = self.project_root / 'config' / 'tier_metadata.json'

        # Load tier metadata for order and titles
        tier_order = []
        tier_titles = {}
        if config_path.exists():
            try:
                meta = json.loads(config_path.read_text(encoding='utf-8'))
                tier_order = meta.get('order', [])
                tiers = meta.get('tiers', {})
                for key, info in tiers.items():
                    title = info.get('title', key)
                    desc = info.get('description')
                    tier_titles[key] = f"{title} ({desc})" if desc else title
            except Exception:
                pass

        # Discover tiers
        discovered = [p.name for p in results_dir.iterdir() if p.is_dir()]
        # Ignore non-tier and temporary/demo tiers
        ignore = {'.git', '__pycache__', 'Tier6', 'Tier6Demo'}
        discovered = [d for d in discovered if d not in ignore]
        ordered_tiers = [t for t in tier_order if t in discovered] + [t for t in discovered if t not in tier_order]

        # Aggregate pass/fail counts per tier
        rows = []
        tier_details = []
        for tier in ordered_tiers:
            tier_path = results_dir / tier
            tests = [d for d in tier_path.iterdir() if d.is_dir()]
            total = 0
            passed = 0
            for tdir in tests:
                s = tdir / 'summary.json'
                total += 1
                if s.exists():
                    try:
                        data = json.loads(s.read_text(encoding='utf-8'))
                        p = data.get('passed')
                        if p is True or str(p).lower() in ['true', 'pass', 'passed']:
                            passed += 1
                    except Exception:
                        pass
            title = tier_titles.get(tier, tier)
            key = 'Key achievements available in tier descriptions'
            pr = f"{passed}/{total} ({(passed/total*100 if total else 0):.0f}%)"
            rows.append((title, tier, total, pr, key))

            # Collect per-tier detail section
            tier_details.append(f"## {title}")
            tier_details.append('')
            # List tests with status and description
            for tdir in sorted(tests):
                tid = tdir.name
                desc = 'No description available'
                status = 'UNKNOWN'
                s = tdir / 'summary.json'
                r = tdir / 'readme.txt'
                if s.exists():
                    try:
                        data = json.loads(s.read_text(encoding='utf-8'))
                        desc = data.get('description', desc)
                        p = data.get('passed')
                        if p is True:
                            status = 'PASS'
                        elif p is False:
                            status = 'FAIL'
                    except Exception:
                        pass
                if r.exists():
                    try:
                        txt = r.read_text(encoding='utf-8')
                        # Use first non-empty line as a friendly description fallback
                        for line in txt.splitlines():
                            if line.strip():
                                desc = desc if desc != 'No description available' else line.strip()
                                break
                    except Exception:
                        pass
                tier_details.append(f"- {tid}: {status} — {desc}")
            tier_details.append('')

        # Build markdown
        lines = ["# Comprehensive Test Results", "", "## Overview", "", 
                 "This report is generated directly from the results/* directories. As tests are added or updated, this section automatically reflects the current state.", "",
                 "## Test Summary", "", "| Tier | Category | Tests | Pass Rate | Notes |", "|------|----------|--------|-----------|-------|"]

        for title, tier, total, pr, key in rows:
            lines.append(f"| {title.split(' — ')[0]} | {tier} | {total} | {pr} | {key} |")

        lines.append('')
        lines.append('---')
        lines.append('')
        lines.extend(tier_details)
        lines.append('')
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text('\n'.join(lines), encoding='utf-8')
        return True

    def _generate_evidence_review(self, dest_path: Path, filename: str) -> bool:
        """Run the docs consistency checker to generate EVIDENCE_REVIEW.md"""
        checker = self.project_root / 'tools' / 'docs_consistency_checker.py'
        try:
            if not checker.exists():
                print("⚠️  Evidence checker not found; skipping review generation")
                return False
            result = subprocess.run([
                sys.executable, str(checker)
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)

            # The checker writes directly to docs/upload/EVIDENCE_REVIEW.md
            # Ensure the file exists at the expected destination
            if not dest_path.exists():
                # Try to locate generated file to copy
                generated = self.upload_dir / 'EVIDENCE_REVIEW.md'
                if generated.exists():
                    shutil.copy2(generated, dest_path)
                else:
                    # Nothing was produced
                    return False

            if result.returncode != 0 and self.enforce_review:
                # Treat critical review issues as build failure only if enforcement is enabled
                print("❌ Evidence review found critical issues and enforcement is enabled")
                print(result.stdout)
                print(result.stderr)
                return False

            # Log summary line
            print("✓ Evidence review generated")
            if result.stdout:
                # Print a brief hint that review exists
                print("   (review report created; open EVIDENCE_REVIEW.md)")
            return True
        except Exception as e:
            print(f"⚠️  Evidence review generation failed: {e}")
            return False
    
    def _generate_electromagnetic_achievements(self, dest_path: Path, filename: str) -> bool:
        """Generate electromagnetic achievements document from results/Electromagnetic/*"""
        em_dir = self.project_root / 'results' / 'Electromagnetic'
        total = 0
        passed = 0
        bullets: list[str] = []
        if em_dir.exists():
            for tdir in sorted([d for d in em_dir.iterdir() if d.is_dir()]):
                total += 1
                s = tdir / 'summary.json'
                desc = None
                status = None
                if s.exists():
                    try:
                        data = json.loads(s.read_text(encoding='utf-8'))
                        desc = data.get('description')
                        p = data.get('passed')
                        if p is True:
                            passed += 1
                            status = 'PASS'
                        elif p is False:
                            status = 'FAIL'
                    except Exception:
                        pass
                if not desc:
                    r = tdir / 'readme.txt'
                    if r.exists():
                        try:
                            txt = r.read_text(encoding='utf-8').strip().splitlines()
                            if txt:
                                desc = txt[0]
                        except Exception:
                            pass
                desc = desc or 'No description available'
                bullets.append(f"- {tdir.name}: {status or 'UNKNOWN'} — {desc}")

        header = self._generate_markdown_header("Electromagnetic Theory Validation - EM-Analogous Phenomena from Klein-Gordon Framework")
        lines = [header, "", "## Overview", "",
                 "This document is generated directly from results/Electromagnetic; it reflects the current test set without manual edits.",
                 "", "## Test Results Summary", ""]
        rate = f"{passed}/{total} ({(passed/total*100 if total else 0):.0f}%)"
        lines.append(f"**Tier 5 Electromagnetic Tests — Pass rate: {rate}**")
        lines.append('')
        lines.append('## Test Details')
        lines.append('')
        lines.extend(bullets or ['(No EM tests found)'])
        lines.append('')
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text('\n'.join(lines), encoding='utf-8')
        return True
    
    def _generate_comprehensive_pdf(self, dest_path: Path, filename: str) -> bool:
        """Generate comprehensive PDF using existing builder"""
        try:
            # Rename the generated file to match schema expectation
            pdf_script = self.project_root / 'tools' / 'build_comprehensive_pdf.py'
            if not pdf_script.exists():
                return False
            
            result = subprocess.run([
                sys.executable, str(pdf_script)
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Find the generated PDF and copy to expected location
                generated_pdfs = list(self.upload_dir.glob('LFM_Comprehensive_Report_*.pdf'))
                if generated_pdfs:
                    shutil.copy2(generated_pdfs[0], dest_path)
                    # Remove the timestamped version
                    generated_pdfs[0].unlink()
                    return True
            
            return False
        except Exception:
            return False

    def _generate_pdf_from_text(self, dest_path: Path, filename: str) -> bool:
        """Generate a PDF from a text source using pandoc. Uses per-file header if present.
        Relies on self.schema to find the source_path for this filename."""
        try:
            meta = self.schema.files.get(filename)
            if not meta or not meta.source or not meta.source.source_path:
                print(f"⚠️  No source_path for {filename}")
                return False

            source_file = self.project_root / meta.source.source_path
            if not source_file.exists():
                print(f"⚠️  Source text not found for PDF: {source_file}")
                return False

            # Preprocess templates
            raw = source_file.read_text(encoding='utf-8')
            body = self.template_engine.render(raw)

            # Compose a temporary markdown with standard header + per-file header include
            title = filename.replace('.pdf', '').replace('_', ' ')
            md_header = self._generate_markdown_header(title)

            # Try per-file textual header (docs/evidence/docx_headers/<name>_header.txt)
            base = source_file.stem  # e.g., Executive_Summary
            header_txt = self.project_root / 'docs' / 'evidence' / 'docx_headers' / f"{base}_header.txt"
            standard_header = self.project_root / 'docs' / 'templates' / 'standard_header.md'
            standard_footer = self.project_root / 'docs' / 'templates' / 'standard_footer.md'

            header_block = ''
            if header_txt.exists():
                try:
                    header_block = header_txt.read_text(encoding='utf-8') + "\n\n"
                except Exception:
                    header_block = ''
            elif standard_header.exists():
                header_block = standard_header.read_text(encoding='utf-8') + "\n\n"

            footer_block = ''
            if standard_footer.exists():
                try:
                    footer_block = "\n\n" + standard_footer.read_text(encoding='utf-8')
                except Exception:
                    footer_block = ''

            tmp_md = md_header + "\n\n" + header_block + body + footer_block

            # Write temp md
            tmp_dir = self.upload_dir / '_tmp'
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_md_path = tmp_dir / (Path(filename).stem + '.md')
            tmp_md_path.write_text(tmp_md, encoding='utf-8')

            # Try direct PDF via pandoc (if available), else fall back to DOCX->PDF
            pdf_out = dest_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Attempt pandoc -> PDF
            try:
                r = subprocess.run([
                    'pandoc', str(tmp_md_path), '-o', str(pdf_out), '--pdf-engine=xelatex'
                ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and pdf_out.exists() and pdf_out.stat().st_size > 1000:
                    return True
            except Exception:
                pass

            # Fallback: pandoc -> DOCX then docx2pdf
            tmp_docx = tmp_dir / (Path(filename).stem + '.docx')
            r1 = subprocess.run([
                'pandoc', str(tmp_md_path), '-o', str(tmp_docx)
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            if r1.returncode != 0 or not tmp_docx.exists():
                print(f"⚠️  pandoc DOCX conversion failed for {filename}: {r1.stderr[:200] if r1.stderr else ''}")
                # Final fallback: copy legacy PDF if available
                legacy_pdf = self.project_root / 'docs' / 'evidence' / 'pdf' / filename
                if legacy_pdf.exists():
                    print(f"⚠️  Using legacy PDF fallback for {filename}")
                    shutil.copy2(legacy_pdf, pdf_out)
                    return True
                return False

            try:
                import docx2pdf  # type: ignore
                from docx2pdf import convert
                convert(str(tmp_docx), str(pdf_out))
                return pdf_out.exists() and pdf_out.stat().st_size > 1000
            except Exception:
                print("⚠️  docx2pdf not available; attempting legacy PDF fallback")
                legacy_pdf = self.project_root / 'docs' / 'evidence' / 'pdf' / filename
                if legacy_pdf.exists():
                    shutil.copy2(legacy_pdf, pdf_out)
                    print(f"✓ Copied legacy PDF for {filename}")
                    return True
                return False
        except Exception as e:
            print(f"❌ Error generating PDF {filename}: {e}")
            return False
    
    def _generate_plots_overview(self, dest_path: Path, filename: str) -> bool:
        """Generate plots overview document"""
        content = self._generate_markdown_header("Plots and Visualizations Overview") + """

## Representative Scientific Visualizations

This package includes key plots that demonstrate the major theoretical achievements of the Lattice-Field Medium framework.

### Included Plots

#### Relativistic Physics
- **relativistic_dispersion.png**: Demonstrates proper relativistic dispersion relations emerging from the discrete lattice

#### Quantum Mechanics  
- **quantum_interference.png**: Shows quantum interference patterns in double-slit experiments
- **quantum_bound_states.png**: Visualizes quantum bound state formation in potential wells

### Plot Generation

All plots are generated from actual simulation data using the LFM computational framework. Each visualization represents validated theoretical predictions that match analytical expectations.

### Additional Visualizations

The complete results directory contains extensive additional plots covering:
- Gravitational field analogues
- Energy conservation demonstrations  
- Electromagnetic field visualizations
- Quantization emergence plots

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding='utf-8')
        return True

    def _generate_discoveries_overview(self, dest_path: Path, filename: str) -> bool:
        """Generate a summary of discoveries from the registry."""
        reg = self.project_root / 'docs' / 'discoveries' / 'discoveries.json'
        try:
            entries = []
            if reg.exists():
                entries = json.loads(reg.read_text(encoding='utf-8'))
                if not isinstance(entries, list):
                    entries = []
        except Exception:
            entries = []

        title = "Scientific Discoveries and Domains of Emergence"
        header = self._generate_markdown_header(title)
        lines = [header, '', '## Summary Table', '']
        # Build table
        rows = ["| Date | Tier | Title | Evidence |", "|------|------|-------|----------|"]
        for e in sorted(entries, key=lambda x: x.get('date','')):
            date = (e.get('date','') or '')[:10]
            tier = e.get('tier','')
            title = e.get('title','')
            ev = e.get('evidence','')
            rows.append(f"| {date} | {tier} | {title} | {ev} |")
        if len(rows) == 2:
            rows.append('| - | - | (No discoveries recorded) | - |')
        lines.extend(rows)
        lines.append('')
        lines.append('## Detailed List')
        lines.append('')
        for e in sorted(entries, key=lambda x: x.get('date','')):
            lines.append(f"- {e.get('date','')[:10]} — {e.get('title','')} ({e.get('tier','')})")
            if e.get('summary'):
                lines.append(f"  - {e['summary']}")
            if e.get('evidence'):
                lines.append(f"  - Evidence: {e['evidence']}")
            if e.get('links'):
                lines.append(f"  - Links: {', '.join(e['links'])}")
        lines.append('')
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text('\n'.join(lines), encoding='utf-8')
        return True
    
    def _generate_manifest(self, dest_path: Path, filename: str) -> bool:
        """Generate complete file manifest with checksums"""
        content = self._generate_markdown_header("Upload Package Manifest") + """

## File Inventory

This manifest provides a complete inventory of all files in the upload package with SHA-256 checksums for integrity verification.

### Common Files (Both Zenodo and OSF)

| File | Size (bytes) | SHA-256 Checksum | Description |
|------|--------------|------------------|-------------|"""
        
        # Generate manifest for all existing files
        common_files = []
        for file_path in self.upload_dir.rglob('*'):
            if file_path.is_file() and file_path.name != 'MANIFEST.md':
                rel_path = file_path.relative_to(self.upload_dir)
                size = file_path.stat().st_size
                checksum = self._calculate_sha256(file_path)
                common_files.append((str(rel_path), size, checksum))
        
        # Sort and add to manifest
        for file_path, size, checksum in sorted(common_files):
            content += f"\n| `{file_path}` | {size:,} | `{checksum}` | Generated file |"
        
        content += f"""

### Package Statistics

- **Total Files**: {len(common_files)}
- **Total Size**: {sum(f[1] for f in common_files):,} bytes
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Integrity Verification

To verify file integrity, recalculate SHA-256 checksums and compare with values listed above.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding='utf-8')
        return True
    
    def _generate_zenodo_metadata(self, dest_path: Path, filename: str) -> bool:
        """Generate Zenodo metadata JSON"""
        metadata = {
            "title": "Lattice-Field Medium: Complete Electromagnetic Theory Validation",
            "creators": [
                {
                    "name": "Partin, Greg D.",
                    "orcid": "0009-0004-0327-6528",
                    "affiliation": "LFM Research, Los Angeles CA USA"
                }
            ],
            "description": "Validation of electromagnetic theory within the Lattice-Field Medium framework, demonstrating EM-analogous phenomena (wave propagation, field coupling, polarization, birefringence) from Klein-Gordon equation with spatially-varying χ-field. See RESULTS_COMPREHENSIVE.md for current pass rates across tiers, including Tier 5 electromagnetic validations covering wave dynamics, field interactions, and rainbow lensing phenomena.",
            "keywords": [
                "electromagnetic theory",
                "EM-analogous phenomena",
                "lattice field theory",
                "discrete spacetime",
                "Klein-Gordon equation",
                "rainbow lensing",
                "computational physics"
            ],
            "license": "cc-by-nc-nd-4.0",
            "communities": [{"identifier": "physics"}],
            "upload_type": "dataset",
            "publication_date": datetime.now().strftime('%Y-%m-%d'),
            "version": "1.0.0"
        }
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        return True
    
    def _generate_osf_metadata(self, dest_path: Path, filename: str) -> bool:
        """Generate OSF metadata JSON"""
        metadata = {
            "title": "LFM Electromagnetic Theory Dataset",
            "description": "Comprehensive EM-analogous phenomena validation dataset for the Lattice-Field Medium framework, demonstrating electromagnetic behavior from Klein-Gordon dynamics.",
            "category": "physics",
            "tags": ["electromagnetic", "EM-analogous", "discrete spacetime", "lattice field", "Klein-Gordon"],
            "contributors": [
                {
                    "name": "Greg D. Partin",
                    "email": "latticefieldmediumresearch@gmail.com"
                }
            ],
            "license": "CC BY-NC-ND 4.0"
        }
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        return True
    
    def _create_zenodo_bundle(self, dest_path: Path, filename: str) -> bool:
        """No longer needed - Zenodo folder contains flat files ready for upload"""
        print(f"⚠️  ZIP bundle generation deprecated - zenodo/ folder contains flat files")
        return True
    
    def _create_osf_bundle(self, dest_path: Path, filename: str) -> bool:
        """No longer needed - OSF folder contains flat files ready for upload"""
        print(f"⚠️  ZIP bundle generation deprecated - osf/ folder contains flat files")
        return True
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_build_order(self, files: Dict[str, FileMetadata]) -> List[str]:
        """Get correct build order based on dependencies"""
        # Simple dependency resolution for now
        no_deps = []
        with_deps = []
        
        for filename, metadata in files.items():
            if not metadata.source.dependencies or metadata.source.dependencies == ['*']:
                with_deps.append(filename)
            else:
                no_deps.append(filename)
        
        # Ensure MANIFEST is last
        if 'MANIFEST.md' in with_deps:
            with_deps.remove('MANIFEST.md')
            with_deps.append('MANIFEST.md')
        
        return no_deps + with_deps
    
    def _validate_output(self) -> bool:
        """Validate generated output against schema"""
        print("🔍 Validating generated output...")
        
        validator = UploadValidator(self.upload_dir, self.schema)
        results = validator.validate_all()
        
        # Count issues
        errors = [r for r in results if not r.passed and r.severity == 'error']
        warnings = [r for r in results if not r.passed and r.severity == 'warning']
        
        if errors:
            print(f"❌ Validation failed with {len(errors)} errors")
            for result in errors[:5]:  # Show first 5 errors
                print(f"  • {result.message}")
            return False
        elif warnings:
            print(f"⚠️  Validation passed with {len(warnings)} warnings")
            return True
        else:
            print("✅ Validation passed with no issues")
            return True
    
    def _generate_final_bundles(self, platforms: List[str]):
        """Generate final platform bundles"""
        print("📦 Generating final bundles...")
        
        for platform in platforms:
            bundle_path = self.upload_dir / platform / f"{platform.upper()}_UPLOAD.zip"
            success = self._create_platform_bundle(bundle_path, platform)
            if success:
                size = bundle_path.stat().st_size
                print(f"✓ Created {platform} bundle: {size:,} bytes")
    
    def _count_generated_files(self) -> int:
        """Count total generated files"""
        return len([f for f in self.upload_dir.rglob('*') if f.is_file()])

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Metadata-Driven Upload Builder')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing upload')
    parser.add_argument('--platform', choices=['zenodo', 'osf'], action='append', help='Build for specific platform(s)')
    parser.add_argument('--upload-dir', default='docs/upload', help='Upload directory path')
    parser.add_argument('--enforce-review', action='store_true', help='Fail build if evidence review finds critical issues')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    upload_dir = project_root / args.upload_dir
    
    platforms = args.platform or ['zenodo', 'osf']
    
    # Create builder
    builder = MetadataDrivenBuilder(project_root, upload_dir, enforce_review=args.enforce_review)
    
    if args.validate_only:
        # Validation only mode
        if not upload_dir.exists():
            print(f"❌ Upload directory not found: {upload_dir}")
            return False
        
        validator = UploadValidator(upload_dir, builder.schema)
        results = validator.validate_all()
        return validator.print_summary()
    else:
        # Full build mode
        return builder.build_complete_package(platforms)

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)