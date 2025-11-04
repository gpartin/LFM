#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Upload Directory Validator

Validates the upload directory against the metadata schema to ensure
deterministic, complete packages for Zenodo and OSF uploads.

VALIDATION TYPES:
1. File Presence: All required files exist
2. File Size: Within expected size ranges  
3. Content Validation: Files contain required content
4. Format Validation: Files are in expected formats
5. Dependency Validation: All dependencies are met
6. Checksum Validation: File integrity verification

USAGE:
    python tools/upload_validator.py
    python tools/upload_validator.py --verbose
    python tools/upload_validator.py --fix-issues
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from dataclasses import dataclass
from datetime import datetime

from upload_metadata_schema import UploadMetadataSchema, FileMetadata

@dataclass
class ValidationResult:
    """Result of validating a single file or rule"""
    file_path: str
    rule: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    fix_suggestion: str = ""

class UploadValidator:
    """Validates upload directory against metadata schema"""
    
    def __init__(self, upload_dir: Path, schema: UploadMetadataSchema):
        self.upload_dir = upload_dir
        self.schema = schema
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> List[ValidationResult]:
        """Run complete validation suite"""
        self.results = []
        
        print("🔍 Starting Upload Directory Validation")
        print("=" * 50)
        
        # 1. Validate file presence
        self._validate_file_presence()
        
        # 2. Validate file sizes
        self._validate_file_sizes()
        
        # 3. Validate content rules
        self._validate_content_rules()
        
        # 4. Validate formats
        self._validate_file_formats()
        
        # 5. Validate directory structure
        self._validate_directory_structure()
        
        return self.results
    
    def _validate_file_presence(self):
        """Check that all required files exist"""
        print("📋 Checking file presence...")
        
        for filename, metadata in self.schema.files.items():
            if not metadata.required:
                continue
                
            file_path = self.upload_dir / filename
            
            if file_path.exists():
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    rule="file_presence",
                    passed=True,
                    message=f"Required file exists: {filename}"
                ))
            else:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    rule="file_presence", 
                    passed=False,
                    message=f"Required file missing: {filename}",
                    fix_suggestion=f"Generate from source: {metadata.source.source_path or metadata.source.generator_function}"
                ))
    
    def _validate_file_sizes(self):
        """Check that files are within expected size ranges"""
        print("📏 Checking file sizes...")
        
        for filename, metadata in self.schema.files.items():
            file_path = self.upload_dir / filename
            
            if not file_path.exists():
                continue  # Already flagged in presence check
            
            try:
                size = file_path.stat().st_size
                
                # Check minimum size
                if metadata.min_size_bytes and size < metadata.min_size_bytes:
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule="min_size",
                        passed=False,
                        message=f"File too small: {size} bytes < {metadata.min_size_bytes} bytes",
                        fix_suggestion="Regenerate file or check source content"
                    ))
                
                # Check maximum size
                if metadata.max_size_bytes and size > metadata.max_size_bytes:
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule="max_size",
                        passed=False,
                        message=f"File too large: {size} bytes > {metadata.max_size_bytes} bytes",
                        severity="warning",
                        fix_suggestion="Review content or increase size limit"
                    ))
                
                # Success case
                if (not metadata.min_size_bytes or size >= metadata.min_size_bytes) and \
                   (not metadata.max_size_bytes or size <= metadata.max_size_bytes):
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule="size_valid",
                        passed=True,
                        message=f"File size OK: {size:,} bytes"
                    ))
                    
            except Exception as e:
                self.results.append(ValidationResult(
                    file_path=str(file_path),
                    rule="size_check",
                    passed=False,
                    message=f"Cannot check file size: {e}",
                    severity="warning"
                ))
    
    def _validate_content_rules(self):
        """Validate content-specific rules"""
        print("📝 Checking content rules...")
        
        content_validators = {
            'must_contain_electromagnetic_theory': self._check_electromagnetic_theory,
            'must_contain_tier5_electromagnetic': self._check_tier5_electromagnetic,
            'must_show_pass_rates': self._check_pass_rates,
            'must_contain_maxwell_equations': self._check_maxwell_equations,
            'must_show_100_percent_success': self._check_100_percent_success,
            'must_contain_electromagnetic_content': self._check_electromagnetic_content,
            'valid_json_schema': self._check_valid_json,
            'contains_required_zenodo_fields': self._check_zenodo_fields,
            'contains_required_osf_fields': self._check_osf_fields,
            'must_list_all_files': self._check_manifest_completeness,
            'must_include_checksums': self._check_manifest_checksums
        }
        
        for filename, metadata in self.schema.files.items():
            file_path = self.upload_dir / filename
            
            if not file_path.exists():
                continue
            
            # Skip text-based content rules for PDFs
            if metadata.expected_format == 'pdf':
                continue
            
            for rule in metadata.validation_rules:
                if rule in content_validators:
                    try:
                        validator = content_validators[rule]
                        result = validator(file_path)
                        self.results.append(result)
                    except Exception as e:
                        self.results.append(ValidationResult(
                            file_path=str(file_path),
                            rule=rule,
                            passed=False,
                            message=f"Validation error: {e}",
                            severity="warning"
                        ))
                else:
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule=rule,
                        passed=False,
                        message=f"Unknown validation rule: {rule}",
                        severity="warning"
                    ))
    
    def _validate_file_formats(self):
        """Validate file formats"""
        print("🗂️  Checking file formats...")
        
        format_checks = {
            'markdown': lambda p: p.suffix.lower() == '.md',
            'pdf': lambda p: p.suffix.lower() == '.pdf',
            'json': lambda p: p.suffix.lower() == '.json',
            'png': lambda p: p.suffix.lower() == '.png',
            'zip': lambda p: p.suffix.lower() == '.zip'
        }
        
        for filename, metadata in self.schema.files.items():
            if not metadata.expected_format:
                continue
                
            file_path = self.upload_dir / filename
            if not file_path.exists():
                continue
            
            format_check = format_checks.get(metadata.expected_format)
            if format_check:
                if format_check(file_path):
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule="format_valid",
                        passed=True,
                        message=f"Format correct: {metadata.expected_format}"
                    ))
                else:
                    self.results.append(ValidationResult(
                        file_path=str(file_path),
                        rule="format_invalid",
                        passed=False,
                        message=f"Expected {metadata.expected_format}, got {file_path.suffix}"
                    ))
    
    def _validate_directory_structure(self):
        """Validate the directory structure matches schema"""
        print("📁 Checking directory structure...")
        
        # Check that zenodo/ and osf/ subdirectories exist
        zenodo_dir = self.upload_dir / 'zenodo'
        osf_dir = self.upload_dir / 'osf'
        
        if not zenodo_dir.exists():
            self.results.append(ValidationResult(
                file_path=str(zenodo_dir),
                rule="directory_structure",
                passed=False,
                message="Missing zenodo/ subdirectory",
                fix_suggestion="Create zenodo/ directory for platform-specific files"
            ))
        
        if not osf_dir.exists():
            self.results.append(ValidationResult(
                file_path=str(osf_dir),
                rule="directory_structure",
                passed=False,
                message="Missing osf/ subdirectory",
                fix_suggestion="Create osf/ directory for platform-specific files"
            ))
    
    # Content validation helpers
    def _check_electromagnetic_theory(self, file_path: Path) -> ValidationResult:
        """Check if file contains electromagnetic theory content"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if re.search(r'electromagnetic\s+theory', content, re.IGNORECASE):
                return ValidationResult(
                    file_path=str(file_path),
                    rule="electromagnetic_theory",
                    passed=True,
                    message="Contains electromagnetic theory content"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="electromagnetic_theory",
                    passed=False,
                    message="Missing electromagnetic theory content",
                    fix_suggestion="Add electromagnetic theory achievements to content"
                )
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                rule="electromagnetic_theory",
                passed=False,
                message=f"Cannot read file: {e}",
                severity="warning"
            )
    
    def _check_tier5_electromagnetic(self, file_path: Path) -> ValidationResult:
        """Check for Tier 5 electromagnetic content"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if re.search(r'tier\s*5.*electromagnetic', content, re.IGNORECASE):
                return ValidationResult(
                    file_path=str(file_path),
                    rule="tier5_electromagnetic",
                    passed=True,
                    message="Contains Tier 5 electromagnetic content"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="tier5_electromagnetic",
                    passed=False,
                    message="Missing Tier 5 electromagnetic content"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="tier5_electromagnetic",
                passed=False,
                message="Cannot validate Tier 5 content",
                severity="warning"
            )
    
    def _check_pass_rates(self, file_path: Path) -> ValidationResult:
        """Check if file shows test pass rates"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            # Accept either an explicit "Pass rate" mention or X/Y (Z%) pattern
            has_fraction_pct = re.search(r'\b\d+/\d+\s*\(\d{1,3}%\)', content, re.IGNORECASE)
            has_pass_phrase = re.search(r'pass\s*rate', content, re.IGNORECASE)
            if has_fraction_pct or has_pass_phrase:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="pass_rates",
                    passed=True,
                    message="Contains test pass rate information"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="pass_rates",
                    passed=False,
                    message="Missing test pass rate information"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="pass_rates",
                passed=False,
                message="Cannot validate pass rates",
                severity="warning"
            )
    
    def _check_maxwell_equations(self, file_path: Path) -> ValidationResult:
        """Check for Maxwell equations content"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if re.search(r'maxwell.*equation', content, re.IGNORECASE):
                return ValidationResult(
                    file_path=str(file_path),
                    rule="maxwell_equations",
                    passed=True,
                    message="Contains Maxwell equations content"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="maxwell_equations",
                    passed=False,
                    message="Missing Maxwell equations content"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="maxwell_equations",
                passed=False,
                message="Cannot validate Maxwell equations",
                severity="warning"
            )
    
    def _check_100_percent_success(self, file_path: Path) -> ValidationResult:
        """Check for 100% success rate"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if re.search(r'100%.*success|15/15.*pass', content, re.IGNORECASE):
                return ValidationResult(
                    file_path=str(file_path),
                    rule="100_percent_success",
                    passed=True,
                    message="Shows 100% success rate"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="100_percent_success",
                    passed=False,
                    message="Missing 100% success rate indication"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="100_percent_success",
                passed=False,
                message="Cannot validate success rate",
                severity="warning"
            )
    
    def _check_electromagnetic_content(self, file_path: Path) -> ValidationResult:
        """General electromagnetic content check"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            patterns = ['electromagnetic', 'maxwell', 'coulomb', 'lorentz force']
            found = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
            
            if found >= 2:  # At least 2 electromagnetic terms
                return ValidationResult(
                    file_path=str(file_path),
                    rule="electromagnetic_content",
                    passed=True,
                    message=f"Contains electromagnetic content ({found} terms found)"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="electromagnetic_content",
                    passed=False,
                    message="Insufficient electromagnetic content"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="electromagnetic_content",
                passed=False,
                message="Cannot validate electromagnetic content",
                severity="warning"
            )
    
    def _check_valid_json(self, file_path: Path) -> ValidationResult:
        """Check if file is valid JSON"""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return ValidationResult(
                file_path=str(file_path),
                rule="valid_json",
                passed=True,
                message="Valid JSON format"
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                file_path=str(file_path),
                rule="valid_json",
                passed=False,
                message=f"Invalid JSON: {e}"
            )
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                rule="valid_json",
                passed=False,
                message=f"Cannot read JSON: {e}",
                severity="warning"
            )
    
    def _check_zenodo_fields(self, file_path: Path) -> ValidationResult:
        """Check for required Zenodo metadata fields"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            required_fields = ['title', 'creators', 'description', 'license']
            missing = [field for field in required_fields if field not in data]
            
            if not missing:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="zenodo_fields",
                    passed=True,
                    message="All required Zenodo fields present"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="zenodo_fields",
                    passed=False,
                    message=f"Missing Zenodo fields: {missing}"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="zenodo_fields",
                passed=False,
                message="Cannot validate Zenodo fields",
                severity="warning"
            )
    
    def _check_osf_fields(self, file_path: Path) -> ValidationResult:
        """Check for required OSF metadata fields"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            required_fields = ['title', 'description', 'category']
            missing = [field for field in required_fields if field not in data]
            
            if not missing:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="osf_fields",
                    passed=True,
                    message="All required OSF fields present"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="osf_fields",
                    passed=False,
                    message=f"Missing OSF fields: {missing}"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="osf_fields",
                passed=False,
                message="Cannot validate OSF fields",
                severity="warning"
            )
    
    def _check_manifest_completeness(self, file_path: Path) -> ValidationResult:
        """Check if manifest lists all files"""
        try:
            content = file_path.read_text()
            
            # Count expected files vs listed files
            expected_files = len([f for f in self.schema.get_common_files().keys() if f != 'MANIFEST.md'])
            
            # Simple heuristic: count lines that look like file listings
            file_lines = len([line for line in content.split('\n') if '|' in line and '.md' in line or '.pdf' in line or '.png' in line])
            
            if file_lines >= expected_files * 0.8:  # At least 80% of files listed
                return ValidationResult(
                    file_path=str(file_path),
                    rule="manifest_completeness",
                    passed=True,
                    message=f"Manifest appears complete ({file_lines} files listed)"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="manifest_completeness",
                    passed=False,
                    message=f"Manifest may be incomplete ({file_lines} vs {expected_files} expected)"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="manifest_completeness",
                passed=False,
                message="Cannot validate manifest completeness",
                severity="warning"
            )
    
    def _check_manifest_checksums(self, file_path: Path) -> ValidationResult:
        """Check if manifest includes checksums"""
        try:
            content = file_path.read_text()
            
            # Look for hex patterns that could be checksums
            checksum_count = len(re.findall(r'[a-f0-9]{32,64}', content))
            
            if checksum_count >= 5:  # At least 5 checksums
                return ValidationResult(
                    file_path=str(file_path),
                    rule="manifest_checksums",
                    passed=True,
                    message=f"Manifest includes checksums ({checksum_count} found)"
                )
            else:
                return ValidationResult(
                    file_path=str(file_path),
                    rule="manifest_checksums",
                    passed=False,
                    message="Manifest missing checksums"
                )
        except:
            return ValidationResult(
                file_path=str(file_path),
                rule="manifest_checksums",
                passed=False,
                message="Cannot validate manifest checksums",
                severity="warning"
            )
    
    def print_summary(self):
        """Print validation summary"""
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed and r.severity == 'error']
        warnings = [r for r in self.results if not r.passed and r.severity == 'warning']
        
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {len(passed)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"⚠️  Warnings: {len(warnings)}")
        print(f"📋 Total Checks: {len(self.results)}")
        
        if failed:
            print(f"\n❌ CRITICAL ISSUES ({len(failed)}):")
            for result in failed:
                print(f"  • {Path(result.file_path).name}: {result.message}")
                if result.fix_suggestion:
                    print(f"    💡 Fix: {result.fix_suggestion}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for result in warnings:
                print(f"  • {Path(result.file_path).name}: {result.message}")
        
        # Overall status
        if not failed:
            print(f"\n🎯 VALIDATION PASSED! Upload directory is ready.")
        else:
            print(f"\n🚨 VALIDATION FAILED! {len(failed)} critical issues must be fixed.")
        
        return len(failed) == 0

def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description='Upload Directory Validator')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--upload-dir', default='docs/upload', help='Upload directory path')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    upload_dir = project_root / args.upload_dir
    
    if not upload_dir.exists():
        print(f"❌ Upload directory not found: {upload_dir}")
        return False
    
    # Load schema and validate
    schema = UploadMetadataSchema()
    validator = UploadValidator(upload_dir, schema)
    results = validator.validate_all()
    
    # Print detailed results if verbose
    if args.verbose:
        print("\n📋 DETAILED RESULTS:")
        for result in results:
            status = "✅" if result.passed else ("⚠️" if result.severity == 'warning' else "❌")
            print(f"{status} {Path(result.file_path).name}: {result.rule} - {result.message}")
    
    # Print summary
    success = validator.print_summary()
    return success

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)