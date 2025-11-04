#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Upload Directory Metadata Schema

This module defines the complete metadata schema for the upload directory,
specifying exactly what files should exist, where they come from, and how
to validate the final package for both Zenodo and OSF uploads.

DIRECTORY STRUCTURE:
upload/
├── [COMMON FILES] - Files used by both Zenodo and OSF
├── zenodo/ - Zenodo-specific files (metadata, etc.)
└── osf/ - OSF-specific files (project files, etc.)

METADATA-DRIVEN APPROACH:
1. Define source → output mapping for every file
2. Specify validation criteria for each file
3. Enable deterministic builds and validation
4. Support both Zenodo and OSF requirements
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class FileSource:
    """Defines where a file comes from and how to generate it"""
    source_type: Literal['copy', 'convert', 'generate', 'external']
    source_path: Optional[str] = None  # Relative to project root
    generator_function: Optional[str] = None  # Function name if generated
    dependencies: List[str] = field(default_factory=list)  # Other files this depends on
    validation_rules: List[str] = field(default_factory=list)  # Validation requirements
    description: str = ""

@dataclass
class FileMetadata:
    """Complete metadata for a file in the upload directory"""
    filename: str
    destination: Literal['common', 'zenodo', 'osf']  # Where in upload structure
    source: FileSource
    required: bool = True
    min_size_bytes: Optional[int] = None
    max_size_bytes: Optional[int] = None
    expected_format: Optional[str] = None
    checksum_validation: bool = False
    validation_rules: List[str] = field(default_factory=list)

class UploadMetadataSchema:
    """Complete schema defining the upload directory structure and validation"""
    
    def __init__(self):
        self.schema_version = "1.0.0"
        self.generated_timestamp = datetime.now().isoformat()
        self.files = self._define_file_schema()
    
    def _define_file_schema(self) -> Dict[str, FileMetadata]:
        """Define the complete file schema for upload directory"""
        
        files = {}
        
        # ===== COMMON FILES (both Zenodo and OSF) =====
        
        # Core documentation (Markdown)
        files['README.md'] = FileMetadata(
            filename='README.md',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='README.md',
                description='Master project README'
            ),
            min_size_bytes=1000,
            expected_format='markdown'
        )
        
        files['EXECUTIVE_SUMMARY.md'] = FileMetadata(
            filename='EXECUTIVE_SUMMARY.md',
            destination='common',
            source=FileSource(
                source_type='convert',
                source_path='docs/text/Executive_Summary.txt',
                generator_function='convert_text_to_markdown',
                description='Executive summary converted from master text'
            ),
            min_size_bytes=2000,
            expected_format='markdown',
            validation_rules=['must_contain_electromagnetic_theory']
        )
        
        files['MASTER_DOCUMENT.md'] = FileMetadata(
            filename='MASTER_DOCUMENT.md',
            destination='common',
            source=FileSource(
                source_type='convert',
                source_path='docs/text/LFM_Master.txt',
                generator_function='convert_text_to_markdown',
                description='Master document converted from authoritative text'
            ),
            min_size_bytes=5000,
            expected_format='markdown'
        )
        
        files['CORE_EQUATIONS.md'] = FileMetadata(
            filename='CORE_EQUATIONS.md',
            destination='common',
            source=FileSource(
                source_type='convert',
                source_path='docs/text/LFM_Core_Equations.txt',
                generator_function='convert_text_to_markdown',
                description='Core equations from master equations text'
            ),
            min_size_bytes=3000,
            expected_format='markdown'
        )
        
        files['TEST_DESIGN.md'] = FileMetadata(
            filename='TEST_DESIGN.md',
            destination='common',
            source=FileSource(
                source_type='convert',
                source_path='docs/text/LFM_Phase1_Test_Design.txt',
                generator_function='convert_text_to_markdown',
                description='Test design methodology from master design document'
            ),
            min_size_bytes=2000,
            expected_format='markdown'
        )
        
        # Generated analysis documents
        files['RESULTS_COMPREHENSIVE.md'] = FileMetadata(
            filename='RESULTS_COMPREHENSIVE.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_comprehensive_results',
                dependencies=['results/*/MASTER_TEST_STATUS.csv'],
                description='Comprehensive test results analysis from all tiers'
            ),
            min_size_bytes=3000,
            expected_format='markdown',
            validation_rules=['must_contain_tier5_electromagnetic', 'must_show_pass_rates']
        )
        
        files['ELECTROMAGNETIC_ACHIEVEMENTS.md'] = FileMetadata(
            filename='ELECTROMAGNETIC_ACHIEVEMENTS.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_electromagnetic_achievements',
                dependencies=['results/Electromagnetic/*/MASTER_TEST_STATUS.csv'],
                description='Dedicated electromagnetic theory validation document'
            ),
            min_size_bytes=2000,
            expected_format='markdown',
            validation_rules=['must_show_pass_rates']
        )

        # Evidence review report
        files['EVIDENCE_REVIEW.md'] = FileMetadata(
            filename='EVIDENCE_REVIEW.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_evidence_review',
                dependencies=['docs/text/*.txt', 'results/*/*/summary.json'],
                description='Automated audit comparing evidence docs vs current results'
            ),
            min_size_bytes=500,
            expected_format='markdown'
        )
        
        # PDF documents
        files['Executive_Summary.pdf'] = FileMetadata(
            filename='Executive_Summary.pdf',
            destination='common',
            source=FileSource(
                source_type='generate',
                source_path='docs/text/Executive_Summary.txt',
                generator_function='generate_pdf_from_text',
                dependencies=['docs/templates/standard_header.md', 'docs/templates/standard_footer.md', 'docs/evidence/docx_headers/Executive_Summary_header.txt'],
                description='Executive summary PDF generated from authoritative text with standard header/footer'
            ),
            min_size_bytes=80000,
            max_size_bytes=1000000,
            expected_format='pdf',
            required=False
        )
        
        files['LFM_Master.pdf'] = FileMetadata(
            filename='LFM_Master.pdf',
            destination='common',
            source=FileSource(
                source_type='generate',
                source_path='docs/text/LFM_Master.txt',
                generator_function='generate_pdf_from_text',
                dependencies=['docs/templates/standard_header.md', 'docs/templates/standard_footer.md', 'docs/evidence/docx_headers/LFM_Master_header.txt'],
                description='Master document PDF generated from authoritative text with standard header/footer'
            ),
            min_size_bytes=90000,
            max_size_bytes=2000000,
            expected_format='pdf',
            required=False
        )
        
        files['LFM_Core_Equations.pdf'] = FileMetadata(
            filename='LFM_Core_Equations.pdf',
            destination='common',
            source=FileSource(
                source_type='generate',
                source_path='docs/text/LFM_Core_Equations.txt',
                generator_function='generate_pdf_from_text',
                dependencies=['docs/templates/standard_header.md', 'docs/templates/standard_footer.md', 'docs/evidence/docx_headers/LFM_Core_Equations_header.txt'],
                description='Core equations PDF generated from authoritative text with standard header/footer'
            ),
            min_size_bytes=120000,
            max_size_bytes=2000000,
            expected_format='pdf',
            required=False
        )
        
        files['LFM_Phase1_Test_Design.pdf'] = FileMetadata(
            filename='LFM_Phase1_Test_Design.pdf',
            destination='common',
            source=FileSource(
                source_type='generate',
                source_path='docs/text/LFM_Phase1_Test_Design.txt',
                generator_function='generate_pdf_from_text',
                dependencies=['docs/templates/standard_header.md', 'docs/templates/standard_footer.md', 'docs/evidence/docx_headers/LFM_Phase1_Test_Design_header.txt'],
                description='Test design PDF generated from authoritative text with standard header/footer'
            ),
            min_size_bytes=90000,
            max_size_bytes=2000000,
            expected_format='pdf',
            required=False
        )
        
        # Comprehensive report (generated)
        files['LFM_Comprehensive_Report.pdf'] = FileMetadata(
            filename='LFM_Comprehensive_Report.pdf',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_comprehensive_pdf',
                dependencies=['docs/text/*.txt', 'results/*/MASTER_TEST_STATUS.csv', 'docs/upload/ELECTROMAGNETIC_ACHIEVEMENTS.md'],
                description='Comprehensive combined PDF report'
            ),
            min_size_bytes=200000,
            max_size_bytes=1000000,
            expected_format='pdf',
            required=False
        )
        
        # Legal framework
        files['LICENSE'] = FileMetadata(
            filename='LICENSE',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='LICENSE',
                description='Project license (CC BY-NC-ND 4.0)'
            ),
            min_size_bytes=1000,
            required=True
        )
        
        files['NOTICE'] = FileMetadata(
            filename='NOTICE',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='NOTICE',
                description='Legal notices and attributions'
            ),
            min_size_bytes=500,
            required=True
        )
        
        # Third-party licenses summary (recommended for world-class submissions)
        files['THIRD_PARTY_LICENSES.md'] = FileMetadata(
            filename='THIRD_PARTY_LICENSES.md',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='THIRD_PARTY_LICENSES.md',
                description='Summary of third-party licenses and attributions'
            ),
            min_size_bytes=500,
            expected_format='markdown',
            required=False
        )
        
        # Plots and visualizations
        files['plots/relativistic_dispersion.png'] = FileMetadata(
            filename='plots/relativistic_dispersion.png',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='results/Relativistic/REL-05/plots/concept_REL-05.png',
                description='Key relativistic physics visualization'
            ),
            min_size_bytes=10000,
            expected_format='png',
            required=False  # Optional since plots may not always be available
        )
        
        files['plots/quantum_interference.png'] = FileMetadata(
            filename='plots/quantum_interference.png',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='results/Quantization/QUAN-09/plots/uncertainty_dx_dk.png',
                description='Quantum uncertainty and interference demonstration'
            ),
            min_size_bytes=10000,
            expected_format='png',
            required=False
        )
        
        files['plots/quantum_bound_states.png'] = FileMetadata(
            filename='plots/quantum_bound_states.png',
            destination='common',
            source=FileSource(
                source_type='copy',
                source_path='results/Quantization/QUAN-10/plots/bound_state_modes.png',
                description='Quantum bound states visualization'
            ),
            min_size_bytes=10000,
            expected_format='png',
            required=False
        )
        
        files['PLOTS_OVERVIEW.md'] = FileMetadata(
            filename='PLOTS_OVERVIEW.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_plots_overview',
                dependencies=['plots/*.png'],
                description='Overview of included plots and visualizations'
            ),
            min_size_bytes=500,
            expected_format='markdown'
        )

        # Discoveries overview (registry-driven)
        files['DISCOVERIES_OVERVIEW.md'] = FileMetadata(
            filename='DISCOVERIES_OVERVIEW.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_discoveries_overview',
                dependencies=['docs/discoveries/discoveries.json'],
                description='Overview of recorded discoveries and domains of emergence'
            ),
            min_size_bytes=200,
            expected_format='markdown',
            required=False
        )
        
        # Manifest (must be generated last)
        files['MANIFEST.md'] = FileMetadata(
            filename='MANIFEST.md',
            destination='common',
            source=FileSource(
                source_type='generate',
                generator_function='generate_manifest',
                dependencies=['*'],  # Depends on all other files
                description='Complete manifest of all files with checksums'
            ),
            min_size_bytes=1000,
            expected_format='markdown',
            validation_rules=['must_list_all_files', 'must_include_checksums']
        )
        
        # ===== ZENODO-SPECIFIC FILES =====
        
        files['zenodo/zenodo_metadata.json'] = FileMetadata(
            filename='zenodo/zenodo_metadata.json',
            destination='zenodo',
            source=FileSource(
                source_type='generate',
                generator_function='generate_zenodo_metadata',
                description='Zenodo-specific metadata for academic repository'
            ),
            min_size_bytes=1000,
            expected_format='json',
            validation_rules=['valid_json_schema', 'contains_required_zenodo_fields']
        )
        
        files['zenodo/ZENODO_UPLOAD.zip'] = FileMetadata(
            filename='zenodo/ZENODO_UPLOAD.zip',
            destination='zenodo',
            source=FileSource(
                source_type='generate',
                generator_function='create_zenodo_bundle',
                dependencies=['*'],  # All common files
                description='Complete ZIP bundle for Zenodo upload'
            ),
            min_size_bytes=300000,
            expected_format='zip'
        )
        
        # ===== OSF-SPECIFIC FILES =====
        
        files['osf/osf_project.json'] = FileMetadata(
            filename='osf/osf_project.json',
            destination='osf',
            source=FileSource(
                source_type='generate',
                generator_function='generate_osf_metadata',
                description='OSF-specific project metadata'
            ),
            min_size_bytes=500,
            expected_format='json',
            validation_rules=['valid_json_schema', 'contains_required_osf_fields']
        )
        
        files['osf/OSF_UPLOAD.zip'] = FileMetadata(
            filename='osf/OSF_UPLOAD.zip',
            destination='osf',
            source=FileSource(
                source_type='generate',
                generator_function='create_osf_bundle',
                dependencies=['*'],  # All common files
                description='Complete ZIP bundle for OSF upload'
            ),
            min_size_bytes=300000,
            expected_format='zip'
        )
        
        return files
    
    def get_common_files(self) -> Dict[str, FileMetadata]:
        """Get all files that go in the common upload directory"""
        return {k: v for k, v in self.files.items() if v.destination == 'common'}
    
    def get_zenodo_files(self) -> Dict[str, FileMetadata]:
        """Get all Zenodo-specific files"""
        return {k: v for k, v in self.files.items() if v.destination == 'zenodo'}
    
    def get_osf_files(self) -> Dict[str, FileMetadata]:
        """Get all OSF-specific files"""
        return {k: v for k, v in self.files.items() if v.destination == 'osf'}
    
    def get_build_order(self) -> List[str]:
        """Get files in correct build order based on dependencies"""
        # Simple dependency resolution - files with no dependencies first
        no_deps = []
        with_deps = []
        
        for filename, metadata in self.files.items():
            if not metadata.source.dependencies or metadata.source.dependencies == ['*']:
                with_deps.append(filename)  # Build last if depends on everything
            else:
                no_deps.append(filename)
        
        # MANIFEST must be last
        if 'MANIFEST.md' in with_deps:
            with_deps.remove('MANIFEST.md')
            with_deps.append('MANIFEST.md')
        
        return no_deps + with_deps
    
    def to_json(self) -> str:
        """Export schema as JSON for external tools"""
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        schema_dict = {
            'schema_version': self.schema_version,
            'generated_timestamp': self.generated_timestamp,
            'files': {k: convert_to_dict(v) for k, v in self.files.items()}
        }
        
        # Convert nested objects
        for filename, file_data in schema_dict['files'].items():
            file_data['source'] = convert_to_dict(file_data['source'])
        
        return json.dumps(schema_dict, indent=2)
    
    def validate_requirements(self) -> List[str]:
        """Validate that all requirements are achievable"""
        issues = []
        
        # Check for circular dependencies
        # Check that required source files exist
        # Check that generator functions are available
        # etc.
        
        return issues

def main():
    """Generate and display the upload metadata schema"""
    schema = UploadMetadataSchema()
    
    print("🗂️  LFM Upload Directory Metadata Schema")
    print("=" * 50)
    print(f"Schema Version: {schema.schema_version}")
    print(f"Generated: {schema.generated_timestamp}")
    print()
    
    print("📁 COMMON FILES (both Zenodo and OSF):")
    for filename, metadata in schema.get_common_files().items():
        print(f"  ✓ {filename}")
        print(f"    Source: {metadata.source.source_type} - {metadata.source.description}")
        if metadata.validation_rules:
            print(f"    Validation: {', '.join(metadata.validation_rules)}")
    
    print("\n🔬 ZENODO-SPECIFIC FILES:")
    for filename, metadata in schema.get_zenodo_files().items():
        print(f"  ✓ {filename}")
        print(f"    Source: {metadata.source.source_type} - {metadata.source.description}")
    
    print("\n📊 OSF-SPECIFIC FILES:")
    for filename, metadata in schema.get_osf_files().items():
        print(f"  ✓ {filename}")
        print(f"    Source: {metadata.source.source_type} - {metadata.source.description}")
    
    print(f"\n📋 TOTAL FILES: {len(schema.files)}")
    print(f"   Common: {len(schema.get_common_files())}")
    print(f"   Zenodo: {len(schema.get_zenodo_files())}")
    print(f"   OSF: {len(schema.get_osf_files())}")
    
    # Export schema
    schema_file = Path(__file__).parent / 'upload_schema.json'
    with open(schema_file, 'w') as f:
        f.write(schema.to_json())
    print(f"\n💾 Schema exported to: {schema_file}")

if __name__ == '__main__':
    main()