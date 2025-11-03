#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Prior Art Documentation Generator
====================================
Creates comprehensive prior art documentation from the LFM codebase.
Establishes public timeline of innovations and technical developments.
Prevents third-party patent claims on disclosed methods.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import hashlib
import re

class PriorArtDocumenter:
    def __init__(self, code_dir):
        self.code_dir = Path(code_dir)
        self.report_dir = self.code_dir / "docs" / "prior_art"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def get_git_info(self, filepath):
        """Get git commit information for a file"""
        try:
            # Get first commit (creation)
            result = subprocess.run([
                'git', 'log', '--format=%H|%ad|%s', '--date=iso', '--reverse', '--', str(filepath)
            ], capture_output=True, text=True, cwd=self.code_dir.parent)
            
            if result.returncode == 0 and result.stdout.strip():
                first_line = result.stdout.strip().split('\n')[0]
                commit_hash, date, message = first_line.split('|', 2)
                
                # Get latest commit
                result2 = subprocess.run([
                    'git', 'log', '-1', '--format=%H|%ad|%s', '--date=iso', '--', str(filepath)
                ], capture_output=True, text=True, cwd=self.code_dir.parent)
                
                if result2.returncode == 0 and result2.stdout.strip():
                    latest_line = result2.stdout.strip()
                    latest_hash, latest_date, latest_message = latest_line.split('|', 2)
                    
                    return {
                        'first_commit': {
                            'hash': commit_hash[:8],
                            'date': date,
                            'message': message
                        },
                        'latest_commit': {
                            'hash': latest_hash[:8],
                            'date': latest_date,
                            'message': latest_message
                        }
                    }
        except:
            pass
        
        return None
    
    def extract_technical_innovations(self, filepath):
        """Extract technical innovations and novel methods from source code"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return []
        
        innovations = []
        
        # Look for novel algorithms and methods
        patterns = [
            (r'class\s+(\w+).*?:', 'Novel class/algorithm'),
            (r'def\s+(\w+).*?:', 'Method implementation'),
            (r'# Novel|# Innovation|# New approach', 'Explicit innovation marker'),
            (r'@staticmethod|@classmethod', 'Advanced implementation pattern'),
            (r'cupy|gpu|cuda', 'GPU acceleration technique'),
            (r'parallel|threading|multiprocess', 'Parallel processing method'),
            (r'optimization|optimize', 'Performance optimization'),
            (r'stability|CFL|convergence', 'Numerical stability technique'),
            (r'boundary|periodic|absorbing', 'Boundary condition handling'),
            (r'leapfrog|integration|solver', 'Numerical integration method'),
        ]
        
        for pattern, description in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                context = self.get_line_context(content, line_num, 3)
                
                innovations.append({
                    'type': description,
                    'pattern': pattern,
                    'line': line_num,
                    'context': context,
                    'match': match.group()
                })
        
        return innovations
    
    def get_line_context(self, content, line_num, context_lines=3):
        """Get surrounding lines for context"""
        lines = content.split('\n')
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context.append(f"{i+1:4d}: {prefix}{lines[i]}")
        
        return '\n'.join(context)
    
    def analyze_file_significance(self, filepath):
        """Determine the significance of a file for prior art"""
        filename = filepath.name.lower()
        
        # Core algorithm files
        if any(x in filename for x in ['equation', 'solver', 'algorithm', 'core']):
            return 'CORE_ALGORITHM', 10
        
        # Novel interface/interaction methods
        if any(x in filename for x in ['gui', 'interface', 'control']):
            return 'USER_INTERFACE', 8
        
        # Performance/optimization innovations
        if any(x in filename for x in ['parallel', 'gpu', 'optimization', 'performance']):
            return 'PERFORMANCE', 9
        
        # Visualization and analysis tools
        if any(x in filename for x in ['visual', 'plot', 'analysis', 'monitor']):
            return 'VISUALIZATION', 7
        
        # Test and validation frameworks
        if any(x in filename for x in ['test', 'validation', 'harness']):
            return 'VALIDATION', 6
        
        # Utility and support functions
        return 'UTILITY', 5
    
    def generate_file_report(self, filepath):
        """Generate prior art report for a single file"""
        rel_path = filepath.relative_to(self.code_dir)
        git_info = self.get_git_info(filepath)
        innovations = self.extract_technical_innovations(filepath)
        category, priority = self.analyze_file_significance(filepath)
        
        # Calculate file hash for integrity
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Get file stats
        stats = filepath.stat()
        file_size = stats.st_size
        mod_time = datetime.fromtimestamp(stats.st_mtime).isoformat()
        
        return {
            'filepath': str(rel_path),
            'category': category,
            'priority': priority,
            'file_hash': file_hash,
            'file_size': file_size,
            'modified': mod_time,
            'git_info': git_info,
            'innovations': innovations,
            'line_count': self.count_lines(filepath),
            'docstring': self.extract_docstring(filepath)
        }
    
    def count_lines(self, filepath):
        """Count lines in file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def extract_docstring(self, filepath):
        """Extract main docstring from Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for module docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                return docstring_match.group(1).strip()
        except:
            pass
        return None
    
    def generate_comprehensive_report(self):
        """Generate complete prior art documentation"""
        print("🔍 Analyzing LFM codebase for prior art documentation...")
        
        # Collect all Python files
        python_files = []
        for pattern in ['*.py', 'tests/*.py', 'tools/*.py', 'archive/*.py']:
            python_files.extend(self.code_dir.glob(pattern))
            python_files.extend(self.code_dir.rglob(pattern))
        
        # Remove duplicates
        python_files = list(set(python_files))
        
        # Generate reports for each file
        file_reports = []
        for filepath in python_files:
            if filepath.is_file():
                report = self.generate_file_report(filepath)
                file_reports.append(report)
                print(f"   ✅ Analyzed {filepath.name}")
        
        # Sort by priority and category
        file_reports.sort(key=lambda x: (-x['priority'], x['category'], x['filepath']))
        
        # Generate summary statistics
        total_lines = sum(r['line_count'] for r in file_reports)
        total_innovations = sum(len(r['innovations']) for r in file_reports)
        categories = {}
        for report in file_reports:
            cat = report['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Create comprehensive report
        prior_art_report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'generator': 'LFM Prior Art Documenter v1.0',
                'author': 'Greg D. Partin',
                'project': 'Lattice Field Medium (LFM)',
                'repository': 'https://github.com/gpartin/LFM',
                'license': 'CC BY-NC-ND 4.0',
                'contact': 'latticefieldmediumresearch@gmail.com'
            },
            'summary': {
                'total_files': len(file_reports),
                'total_lines': total_lines,
                'total_innovations': total_innovations,
                'categories': categories,
                'analysis_scope': 'Complete LFM framework codebase'
            },
            'files': file_reports,
            'legal_notice': {
                'purpose': 'Establish prior art for LFM innovations',
                'patent_prevention': 'This documentation prevents third-party patent claims on disclosed methods',
                'copyright': 'All innovations documented herein are copyrighted by Greg D. Partin',
                'disclosure_date': datetime.now().strftime('%Y-%m-%d'),
                'public_repository': 'https://github.com/gpartin/LFM'
            }
        }
        
        return prior_art_report
    
    def save_reports(self, report):
        """Save prior art reports in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.report_dir / f"prior_art_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        md_file = self.report_dir / f"prior_art_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report(report))
        
        # Save current/latest versions (for easy access)
        latest_json = self.report_dir / "latest_prior_art_report.json"
        latest_md = self.report_dir / "latest_prior_art_report.md"
        
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        with open(latest_md, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report(report))
        
        return json_file, md_file
    
    def generate_markdown_report(self, report):
        """Generate human-readable markdown report"""
        md = f"""# LFM Prior Art Documentation Report

**Generated:** {report['metadata']['generated']}  
**Author:** {report['metadata']['author']}  
**Project:** {report['metadata']['project']}  
**Repository:** {report['metadata']['repository']}  
**License:** {report['metadata']['license']}  

---

## Legal Notice

**PURPOSE:** This document establishes prior art for innovations in the Lattice Field Medium (LFM) framework.

**PATENT PREVENTION:** This public documentation prevents third-party patent claims on the methods, algorithms, and techniques disclosed herein.

**COPYRIGHT:** All innovations documented in this report are copyrighted by Greg D. Partin under CC BY-NC-ND 4.0 license.

**DISCLOSURE DATE:** {report['legal_notice']['disclosure_date']}  
**PUBLIC REPOSITORY:** {report['legal_notice']['public_repository']}

---

## Summary

- **Total Files Analyzed:** {report['summary']['total_files']}
- **Total Lines of Code:** {report['summary']['total_lines']:,}
- **Technical Innovations Identified:** {report['summary']['total_innovations']}
- **Analysis Scope:** {report['summary']['analysis_scope']}

### File Categories
"""
        
        for category, count in report['summary']['categories'].items():
            md += f"- **{category.replace('_', ' ').title()}:** {count} files\n"
        
        md += "\n---\n\n## Detailed File Analysis\n\n"
        
        # Group files by category
        by_category = {}
        for file_report in report['files']:
            cat = file_report['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(file_report)
        
        # Generate sections by category
        for category in sorted(by_category.keys()):
            md += f"### {category.replace('_', ' ').title()} Files\n\n"
            
            for file_report in by_category[category]:
                md += f"#### `{file_report['filepath']}`\n\n"
                md += f"**Priority:** {file_report['priority']}/10  \n"
                md += f"**Lines:** {file_report['line_count']}  \n"
                md += f"**File Hash:** `{file_report['file_hash']}`  \n"
                
                if file_report['git_info']:
                    git = file_report['git_info']
                    md += f"**First Commit:** {git['first_commit']['date']} (`{git['first_commit']['hash']}`)  \n"
                    md += f"**Latest Commit:** {git['latest_commit']['date']} (`{git['latest_commit']['hash']}`)  \n"
                
                if file_report['docstring']:
                    md += f"**Description:** {file_report['docstring'][:200]}{'...' if len(file_report['docstring']) > 200 else ''}  \n"
                
                if file_report['innovations']:
                    md += f"**Technical Innovations:** {len(file_report['innovations'])} identified\n\n"
                    
                    for i, innovation in enumerate(file_report['innovations'][:5]):  # Limit to top 5
                        md += f"  {i+1}. **{innovation['type']}** (Line {innovation['line']})\n"
                        md += f"     ```\n     {innovation['match']}\n     ```\n\n"
                
                md += "---\n\n"
        
        md += f"""
## Verification Information

This prior art report can be independently verified through:

1. **Public Repository:** {report['metadata']['repository']}
2. **Git Commit History:** Complete development timeline available
3. **File Hashes:** Each file includes SHA-256 hash for integrity verification
4. **Timestamped Documentation:** Generated {report['metadata']['generated']}

## Legal Significance

This documentation serves as:

- **Defensive Publication:** Prevents third-party patents on disclosed innovations
- **Prior Art Establishment:** Creates public record of technical development timeline
- **Copyright Evidence:** Demonstrates authorship and creation dates
- **Commercial Protection:** Supports licensing and IP enforcement activities

**For questions or licensing inquiries, contact:** {report['metadata']['contact']}

---

*Generated by LFM Prior Art Documenter v1.0*  
*Copyright (c) 2025 Greg D. Partin. All rights reserved.*
"""
        
        return md

def main():
    """Generate comprehensive prior art documentation"""
    print("=" * 70)
    print("  LFM PRIOR ART DOCUMENTATION GENERATOR")
    print("=" * 70)
    print()
    
    code_dir = Path(__file__).parent
    documenter = PriorArtDocumenter(code_dir)
    
    # Generate comprehensive report
    report = documenter.generate_comprehensive_report()
    
    # Save reports
    json_file, md_file = documenter.save_reports(report)
    
    print(f"\n✅ Prior art documentation generated!")
    print(f"📊 JSON Report: {json_file}")
    print(f"📝 Markdown Report: {md_file}")
    print(f"📂 Report Directory: {documenter.report_dir}")
    
    print(f"\n📈 Summary:")
    print(f"   • {report['summary']['total_files']} files analyzed")
    print(f"   • {report['summary']['total_lines']:,} lines of code")
    print(f"   • {report['summary']['total_innovations']} technical innovations identified")
    
    print(f"\n🛡️ Legal Protection:")
    print(f"   • Prior art established for {datetime.now().strftime('%Y-%m-%d')}")
    print(f"   • Public repository creates immutable timestamp")
    print(f"   • Prevents third-party patent claims on disclosed methods")
    print(f"   • Supports future commercial licensing activities")
    
    print(f"\n🌐 Public Access:")
    print(f"   • Report available at: docs/prior_art/")
    print(f"   • GitHub provides independent timestamp verification")
    print(f"   • File hashes enable integrity verification")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())