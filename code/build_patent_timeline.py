#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Prior Art Automation & Timeline Builder
==========================================
Automatically maintains comprehensive prior art documentation.
Builds immutable timeline of innovations for patent defense.
Creates public record with GitHub commit timestamps.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

class PriorArtTimeline:
    def __init__(self, code_dir):
        self.code_dir = Path(code_dir)
        self.prior_art_dir = self.code_dir / "docs" / "prior_art"
        self.timeline_dir = self.prior_art_dir / "timeline"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)
        
    def get_complete_git_history(self):
        """Extract complete development timeline from git"""
        try:
            # Get all commits with files changed
            result = subprocess.run([
                'git', 'log', '--name-only', '--pretty=format:%H|%ad|%an|%s', '--date=iso'
            ], capture_output=True, text=True, cwd=self.code_dir.parent)
            
            if result.returncode != 0:
                return []
            
            commits = []
            current_commit = None
            
            for line in result.stdout.split('\n'):
                if '|' in line and len(line.split('|')) >= 4:
                    # New commit line
                    parts = line.split('|')
                    current_commit = {
                        'hash': parts[0],
                        'date': parts[1],
                        'author': parts[2],
                        'message': '|'.join(parts[3:]),
                        'files': []
                    }
                    commits.append(current_commit)
                elif line.strip() and current_commit and not line.startswith(' '):
                    # File in this commit
                    current_commit['files'].append(line.strip())
            
            return commits
            
        except Exception as e:
            print(f"Error getting git history: {e}")
            return []
    
    def categorize_innovation(self, filename, message):
        """Categorize the type of innovation based on file and commit message"""
        filename = filename.lower()
        message = message.lower()
        
        categories = {
            'CORE_ALGORITHM': ['equation', 'solver', 'core', 'algorithm', 'chi_field'],
            'PERFORMANCE': ['parallel', 'gpu', 'optimization', 'performance', 'backend'],
            'USER_INTERFACE': ['gui', 'interface', 'control_center', 'console'],
            'NUMERICAL_METHOD': ['integration', 'leapfrog', 'stability', 'numeric'],
            'VISUALIZATION': ['visual', 'plot', 'render', 'display'],
            'VALIDATION': ['test', 'validation', 'harness', 'metrics'],
            'ARCHITECTURE': ['config', 'framework', 'structure', 'system'],
            'PHYSICS_MODEL': ['tier1', 'tier2', 'tier3', 'tier4', 'relativistic', 'gravity', 'quantum']
        }
        
        for category, keywords in categories.items():
            if any(kw in filename or kw in message for kw in keywords):
                return category
        
        return 'UTILITY'
    
    def analyze_technical_significance(self, filepath, commit_info):
        """Determine technical significance of a change"""
        if not filepath.endswith('.py'):
            return 'DOCUMENTATION'
        
        filename = Path(filepath).name
        message = commit_info['message']
        
        # High significance indicators
        high_sig = ['algorithm', 'solver', 'core', 'novel', 'innovation', 'new method', 'breakthrough']
        if any(term in message.lower() or term in filename.lower() for term in high_sig):
            return 'HIGH'
        
        # Medium significance indicators  
        med_sig = ['optimization', 'improvement', 'enhancement', 'feature', 'implementation']
        if any(term in message.lower() for term in med_sig):
            return 'MEDIUM'
        
        return 'LOW'
    
    def build_innovation_timeline(self):
        """Build comprehensive timeline of all innovations"""
        print("📅 Building innovation timeline from git history...")
        
        commits = self.get_complete_git_history()
        timeline = []
        
        for commit in commits:
            commit_date = datetime.fromisoformat(commit['date'].replace(' ', 'T'))
            
            for filepath in commit['files']:
                if filepath.startswith('code/') and filepath.endswith('.py'):
                    innovation_category = self.categorize_innovation(filepath, commit['message'])
                    significance = self.analyze_technical_significance(filepath, commit)
                    
                    timeline_entry = {
                        'timestamp': commit['date'],
                        'commit_hash': commit['hash'][:8],
                        'author': commit['author'],
                        'filepath': filepath,
                        'innovation_category': innovation_category,
                        'significance': significance,
                        'description': commit['message'],
                        'date_iso': commit_date.isoformat(),
                        'legal_timestamp': f"{commit_date.strftime('%Y-%m-%d %H:%M:%S')} (Git commit {commit['hash'][:8]})"
                    }
                    
                    timeline.append(timeline_entry)
        
        # Sort by date (oldest first)
        timeline.sort(key=lambda x: x['date_iso'])
        
        return timeline
    
    def generate_patent_defense_report(self, timeline):
        """Generate report specifically for patent defense"""
        
        # Group by innovation category
        by_category = {}
        for entry in timeline:
            cat = entry['innovation_category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(entry)
        
        # Find first disclosure date for each category
        first_disclosures = {}
        for category, entries in by_category.items():
            first_entry = min(entries, key=lambda x: x['date_iso'])
            first_disclosures[category] = first_entry
        
        report = {
            'metadata': {
                'report_type': 'Patent Defense Timeline',
                'generated': datetime.now().isoformat(),
                'author': 'Greg D. Partin',
                'project': 'Lattice Field Medium (LFM)',
                'repository': 'https://github.com/gpartin/LFM',
                'purpose': 'Establish prior art for patent defense',
                'legal_notice': 'This timeline prevents third-party patent claims on disclosed innovations'
            },
            'summary': {
                'total_innovations': len(timeline),
                'categories_covered': len(by_category),
                'timeline_span': f"{timeline[0]['timestamp']} to {timeline[-1]['timestamp']}",
                'first_disclosure': timeline[0]['timestamp'],
                'latest_disclosure': timeline[-1]['timestamp']
            },
            'first_disclosures': first_disclosures,
            'complete_timeline': timeline,
            'high_priority_innovations': [
                entry for entry in timeline 
                if entry['significance'] == 'HIGH'
            ],
            'patent_blocking_evidence': {
                'public_repository': 'https://github.com/gpartin/LFM',
                'immutable_timestamps': 'Git commit hashes provide cryptographic proof of dates',
                'continuous_development': f"Development spans {len(set(e['timestamp'][:10] for e in timeline))} days",
                'comprehensive_disclosure': 'Complete source code and documentation publicly available'
            }
        }
        
        return report
    
    def save_timeline_reports(self, timeline, defense_report):
        """Save timeline in multiple formats for legal use"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete timeline
        timeline_file = self.timeline_dir / f"innovation_timeline_{timestamp}.json"
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
        
        # Save patent defense report
        defense_file = self.timeline_dir / f"patent_defense_report_{timestamp}.json"
        with open(defense_file, 'w', encoding='utf-8') as f:
            json.dump(defense_report, f, indent=2, ensure_ascii=False)
        
        # Create human-readable timeline
        timeline_md = self.timeline_dir / f"innovation_timeline_{timestamp}.md"
        with open(timeline_md, 'w', encoding='utf-8') as f:
            f.write(self.generate_timeline_markdown(timeline, defense_report))
        
        # Update latest versions
        latest_timeline = self.timeline_dir / "latest_innovation_timeline.json"
        latest_defense = self.timeline_dir / "latest_patent_defense_report.json"
        latest_md = self.timeline_dir / "latest_innovation_timeline.md"
        
        with open(latest_timeline, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
        
        with open(latest_defense, 'w', encoding='utf-8') as f:
            json.dump(defense_report, f, indent=2, ensure_ascii=False)
        
        with open(latest_md, 'w', encoding='utf-8') as f:
            f.write(self.generate_timeline_markdown(timeline, defense_report))
        
        return timeline_file, defense_file, timeline_md
    
    def generate_timeline_markdown(self, timeline, defense_report):
        """Generate comprehensive timeline documentation"""
        md = f"""# LFM Innovation Timeline - Patent Defense Documentation

**Generated:** {defense_report['metadata']['generated']}  
**Author:** {defense_report['metadata']['author']}  
**Repository:** {defense_report['metadata']['repository']}  
**Purpose:** {defense_report['metadata']['purpose']}  

---

## 🛡️ Legal Notice

**PATENT DEFENSE:** This document establishes a complete, immutable timeline of LFM innovations to prevent third-party patent claims.

**PUBLIC DISCLOSURE:** All innovations documented herein have been publicly disclosed via GitHub repository with cryptographically-verified timestamps.

**PRIOR ART ESTABLISHMENT:** These disclosures constitute prior art that invalidates any subsequent patent applications on the same methods.

**COPYRIGHT:** All innovations are copyrighted by Greg D. Partin under CC BY-NC-ND 4.0.

---

## 📊 Timeline Summary

- **Total Innovations Documented:** {defense_report['summary']['total_innovations']}
- **Innovation Categories:** {defense_report['summary']['categories_covered']}
- **Development Timespan:** {defense_report['summary']['timeline_span']}
- **First Public Disclosure:** {defense_report['summary']['first_disclosure']}

---

## 🏆 First Disclosure Dates (Critical for Patent Defense)

"""
        
        for category, entry in defense_report['first_disclosures'].items():
            md += f"### {category.replace('_', ' ').title()}\n"
            md += f"**First Disclosed:** {entry['timestamp']}  \n"
            md += f"**Commit:** `{entry['commit_hash']}`  \n"
            md += f"**File:** `{entry['filepath']}`  \n"
            md += f"**Description:** {entry['description']}  \n\n"
        
        md += "---\n\n## 🔥 High-Priority Innovations\n\n"
        
        for entry in defense_report['high_priority_innovations']:
            md += f"### {entry['timestamp']} - {entry['innovation_category']}\n"
            md += f"**File:** `{entry['filepath']}`  \n"
            md += f"**Commit:** `{entry['commit_hash']}`  \n"
            md += f"**Description:** {entry['description']}  \n"
            md += f"**Legal Timestamp:** {entry['legal_timestamp']}  \n\n"
        
        md += "---\n\n## 📅 Complete Innovation Timeline\n\n"
        
        current_date = None
        for entry in timeline:
            entry_date = entry['timestamp'][:10]
            if entry_date != current_date:
                current_date = entry_date
                md += f"\n### {current_date}\n\n"
            
            md += f"**{entry['timestamp']}** - `{entry['filepath']}`  \n"
            md += f"Category: {entry['innovation_category']} | Significance: {entry['significance']}  \n"
            md += f"Commit: `{entry['commit_hash']}` | {entry['description']}  \n\n"
        
        md += f"""---

## 🔐 Verification & Legal Standing

### Cryptographic Proof
- **Git Commit Hashes:** Each innovation has cryptographic proof of timestamp
- **Public Repository:** {defense_report['metadata']['repository']}
- **Immutable Record:** Git history cannot be altered without detection

### Patent Defense Evidence
- **Public Disclosure:** {defense_report['patent_blocking_evidence']['public_repository']}
- **Timestamp Verification:** {defense_report['patent_blocking_evidence']['immutable_timestamps']}
- **Development History:** {defense_report['patent_blocking_evidence']['continuous_development']}
- **Complete Disclosure:** {defense_report['patent_blocking_evidence']['comprehensive_disclosure']}

### Legal Contact
For patent challenges, licensing inquiries, or legal matters:  
**Contact:** latticefieldmediumresearch@gmail.com

---

*This timeline serves as definitive prior art documentation to prevent third-party patent claims on LFM innovations.*

**Generated by LFM Prior Art Timeline Builder**  
**Copyright (c) 2025 Greg D. Partin. All rights reserved.**
"""
        
        return md

def main():
    """Generate comprehensive prior art timeline for patent defense"""
    print("=" * 70)
    print("  LFM PRIOR ART TIMELINE - PATENT DEFENSE BUILDER")
    print("=" * 70)
    print()
    
    code_dir = Path(__file__).parent
    timeline_builder = PriorArtTimeline(code_dir)
    
    # Build innovation timeline
    timeline = timeline_builder.build_innovation_timeline()
    
    if not timeline:
        print("❌ Could not build timeline - no git history found")
        return 1
    
    # Generate patent defense report
    defense_report = timeline_builder.generate_patent_defense_report(timeline)
    
    # Save all reports
    timeline_file, defense_file, md_file = timeline_builder.save_timeline_reports(timeline, defense_report)
    
    print(f"✅ Innovation timeline generated!")
    print(f"📅 Timeline File: {timeline_file}")
    print(f"🛡️ Defense Report: {defense_file}")
    print(f"📝 Markdown Report: {md_file}")
    
    print(f"\n📈 Timeline Summary:")
    print(f"   • {defense_report['summary']['total_innovations']} innovations documented")
    print(f"   • {defense_report['summary']['categories_covered']} technical categories")
    print(f"   • Development span: {defense_report['summary']['timeline_span']}")
    print(f"   • First disclosure: {defense_report['summary']['first_disclosure']}")
    
    print(f"\n🛡️ Patent Defense Status:")
    print(f"   • Complete public disclosure established")
    print(f"   • Immutable timestamps via Git commits")
    print(f"   • {len(defense_report['high_priority_innovations'])} high-priority innovations protected")
    print(f"   • Prior art prevents third-party patents")
    
    print(f"\n🔗 Public Evidence:")
    print(f"   • Repository: {defense_report['metadata']['repository']}")
    print(f"   • Timeline: docs/prior_art/timeline/")
    print(f"   • Verification: Git commit hashes provide cryptographic proof")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())