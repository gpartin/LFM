#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Version Manager

Manages version numbers and release dates across all LFM documents and metadata files.
Ensures consistency when generating upload packages for defensive publication.

USAGE:
    python tools/version_manager.py --bump minor  # 3.0 -> 3.1
    python tools/version_manager.py --bump major  # 3.1 -> 4.0
    python tools/version_manager.py --set 3.1 --date 2025-11-05
    python tools/version_manager.py --show  # Display current version
    
VERSION FORMAT: X.Y (e.g., 3.0, 3.1, 4.0)
- Major version (X): Fundamental changes to physics equation or framework
- Minor version (Y): Documentation updates, test additions, clarifications

FILES UPDATED:
- docs/text/*.txt (Executive_Summary, LFM_Master, LFM_Core_Equations, LFM_Phase1_Test_Design)
- workspace/VERSION (canonical version file)
- All generated upload documents automatically inherit from VERSION file
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class VersionManager:
    """Manages LFM version numbers and release dates across all documents"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.version_file = workspace_root / 'VERSION'
        self.docs_text_dir = workspace_root / 'docs' / 'text'
        
        # Files to update with version/date
        self.text_files = [
            'Executive_Summary.txt',
            'LFM_Master.txt',
            'LFM_Core_Equations.txt',
            'LFM_Phase1_Test_Design.txt'
        ]
    
    def get_current_version(self) -> Tuple[str, str]:
        """Get current version and date from VERSION file"""
        if not self.version_file.exists():
            # Create default VERSION file
            self._write_version_file('3.0', datetime.now().strftime('%Y-%m-%d'))
            return '3.0', datetime.now().strftime('%Y-%m-%d')
        
        try:
            data = json.loads(self.version_file.read_text(encoding='utf-8'))
            return data['version'], data['date']
        except Exception as e:
            print(f"Error reading VERSION file: {e}")
            return '3.0', datetime.now().strftime('%Y-%m-%d')
    
    def _write_version_file(self, version: str, date: str):
        """Write version and date to VERSION file"""
        data = {
            'version': version,
            'date': date,
            'description': f'LFM Defensive ND Release v{version}',
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.version_file.write_text(
            json.dumps(data, indent=2) + '\n',
            encoding='utf-8'
        )
        print(f"✓ Updated VERSION file: v{version} ({date})")
    
    def bump_version(self, bump_type: str, new_date: str = None) -> Tuple[str, str]:
        """Bump version number (major or minor)"""
        current_version, _ = self.get_current_version()
        
        try:
            major, minor = map(int, current_version.split('.'))
        except ValueError:
            print(f"Error: Invalid version format '{current_version}'. Expected X.Y")
            return current_version, new_date or datetime.now().strftime('%Y-%m-%d')
        
        if bump_type == 'major':
            new_version = f"{major + 1}.0"
        elif bump_type == 'minor':
            new_version = f"{major}.{minor + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}. Use 'major' or 'minor'")
        
        release_date = new_date or datetime.now().strftime('%Y-%m-%d')
        self._write_version_file(new_version, release_date)
        return new_version, release_date
    
    def set_version(self, version: str, date: str = None) -> Tuple[str, str]:
        """Set specific version and date"""
        # Validate version format
        if not re.match(r'^\d+\.\d+$', version):
            raise ValueError(f"Invalid version format: {version}. Expected X.Y (e.g., 3.1)")
        
        release_date = date or datetime.now().strftime('%Y-%m-%d')
        
        # Validate date format
        try:
            datetime.strptime(release_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {release_date}. Expected YYYY-MM-DD")
        
        self._write_version_file(version, release_date)
        return version, release_date
    
    def update_text_files(self, version: str, date: str) -> List[str]:
        """Update version and date in all text documentation files"""
        updated_files = []
        
        for filename in self.text_files:
            filepath = self.docs_text_dir / filename
            if not filepath.exists():
                print(f"⚠️  Skipping missing file: {filename}")
                continue
            
            try:
                content = filepath.read_text(encoding='utf-8')
                original_content = content
                
                # Pattern 1: "Version X.Y — YYYY-MM-DD"
                content = re.sub(
                    r'Version \d+\.\d+ — \d{4}-\d{2}-\d{2}',
                    f'Version {version} — {date}',
                    content
                )
                
                # Pattern 2: "(v3.0 — 2025-11-01)" or similar
                content = re.sub(
                    r'\(v\d+\.\d+ — \d{4}-\d{2}-\d{2}\)',
                    f'(v{version} — {date})',
                    content
                )
                
                # Pattern 3: "last updated YYYY-MM-DD" in discovery references
                # Only update if it's about the document version, not discovery dates
                # We'll be conservative here and not touch these
                
                if content != original_content:
                    filepath.write_text(content, encoding='utf-8')
                    updated_files.append(filename)
                    print(f"✓ Updated {filename}")
                else:
                    print(f"ℹ️  No changes needed in {filename}")
                    
            except Exception as e:
                print(f"❌ Error updating {filename}: {e}")
        
        return updated_files
    
    def generate_release_notes(self, version: str, date: str, changes: List[str] = None) -> str:
        """Generate release notes for this version"""
        notes = [
            f"# LFM Release v{version}",
            f"**Release Date**: {date}",
            f"**Type**: Defensive ND Release",
            "",
            "## Changes in this Release",
            ""
        ]
        
        if changes:
            for change in changes:
                notes.append(f"- {change}")
        else:
            notes.append("- Version and date synchronization across all documents")
            notes.append("- Upload package regenerated with updated metadata")
        
        notes.extend([
            "",
            "## Updated Files",
            ""
        ])
        
        for filename in self.text_files:
            notes.append(f"- docs/text/{filename}")
        
        notes.extend([
            "",
            "## License",
            "CC BY-NC-ND 4.0 International (No Derivatives)",
            "",
            "## Purpose",
            "This release maintains defensive publication status and IP priority claims.",
            ""
        ])
        
        return '\n'.join(notes)
    
    def show_current_version(self):
        """Display current version information"""
        version, date = self.get_current_version()
        
        print("=" * 60)
        print("LFM Version Manager - Current Status")
        print("=" * 60)
        print(f"Version: {version}")
        print(f"Release Date: {date}")
        print(f"Version File: {self.version_file}")
        print("")
        print("Managed Files:")
        for filename in self.text_files:
            filepath = self.docs_text_dir / filename
            status = "✓ Exists" if filepath.exists() else "✗ Missing"
            print(f"  {status} - {filename}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='LFM Version Manager - Synchronize version numbers across all documents'
    )
    parser.add_argument(
        '--bump',
        choices=['major', 'minor'],
        help='Bump version number (major: X.0, minor: X.Y)'
    )
    parser.add_argument(
        '--set',
        metavar='VERSION',
        help='Set specific version (format: X.Y, e.g., 3.1)'
    )
    parser.add_argument(
        '--date',
        metavar='YYYY-MM-DD',
        help='Set specific release date (default: today)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current version information'
    )
    parser.add_argument(
        '--release-notes',
        action='store_true',
        help='Generate release notes for current version'
    )
    
    args = parser.parse_args()
    
    # Find workspace root
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    
    manager = VersionManager(workspace_root)
    
    if args.show:
        manager.show_current_version()
        return 0
    
    if args.release_notes:
        version, date = manager.get_current_version()
        notes = manager.generate_release_notes(version, date)
        print(notes)
        return 0
    
    # Update version
    if args.bump:
        version, date = manager.bump_version(args.bump, args.date)
        print(f"\n✅ Bumped version to v{version} ({date})")
    elif args.set:
        version, date = manager.set_version(args.set, args.date)
        print(f"\n✅ Set version to v{version} ({date})")
    else:
        # No action specified, just show current version
        manager.show_current_version()
        return 0
    
    # Update all text files
    print("\nUpdating documentation files...")
    updated_files = manager.update_text_files(version, date)
    
    print(f"\n✅ Version update complete!")
    print(f"   Updated {len(updated_files)} file(s)")
    print(f"\n💡 Next steps:")
    print(f"   1. Review changes: git diff docs/text/")
    print(f"   2. Regenerate uploads: python tools/build_upload_package.py --deterministic")
    print(f"   3. Commit changes: git add VERSION docs/text/ && git commit -m 'Release v{version}'")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
