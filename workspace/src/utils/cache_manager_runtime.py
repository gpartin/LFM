#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Runtime Test Cache Manager (clean module)

Same API as the build/scripts version, but placed under src/ so all runtime
code can import it without sys.path modifications.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class TestCacheManager:
    """Manages test result caching based on dependency hashing"""

    def __init__(self, cache_root: Path, workspace_root: Path):
        self.cache_root = cache_root
        self.workspace_root = workspace_root
        self.cache_index_path = cache_root / 'cache_index.json'
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict:
        """Load cache index from disk"""
        if self.cache_index_path.exists():
            try:
                return json.loads(self.cache_index_path.read_text(encoding='utf-8'))
            except Exception:
                return {}
        return {}

    def _save_cache_index(self):
        """Save cache index to disk"""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_index_path.write_text(
            json.dumps(self.cache_index, indent=2),
            encoding='utf-8'
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content"""
        if not file_path.exists():
            return "MISSING"

        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return "ERROR"

    def _compute_directory_hash(self, dir_path: Path, patterns: List[str] = None) -> str:
        """Compute combined hash of directory contents"""
        if not dir_path.exists():
            return "MISSING"

        patterns = patterns or ['*.py', '*.json', '*.yaml', '*.txt']
        file_hashes = []

        for pattern in patterns:
            for file_path in sorted(dir_path.rglob(pattern)):
                if file_path.is_file():
                    rel_path = file_path.relative_to(dir_path)
                    file_hash = self._compute_file_hash(file_path)
                    file_hashes.append(f"{rel_path}:{file_hash}")

        combined = '\n'.join(file_hashes)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def compute_test_dependencies_hash(self, test_id: str, config_file: Optional[Path] = None) -> Dict[str, str]:
        """
        Compute hashes for all test dependencies.

        Returns dict of:
        - config_hash: Hash of tier configuration file
        - src_hash: Hash of source code (src/*)
        - test_hash: Hash of test file itself
        - settings_hash: Hash of global settings
        """
        hashes = {}

        # Configuration file hash
        if config_file:
            if isinstance(config_file, Path):
                if config_file.exists() and config_file.is_absolute():
                    # Absolute path provided - use directly
                    hashes['config_hash'] = self._compute_file_hash(config_file)
                elif config_file.exists():
                    # Relative path - resolve against workspace
                    hashes['config_hash'] = self._compute_file_hash(self.workspace_root / config_file)
            else:
                # String path - convert to Path
                config_path = Path(config_file)
                if config_path.exists():
                    hashes['config_hash'] = self._compute_file_hash(config_path)

        # If no config hash yet, try to infer from test_id
        if 'config_hash' not in hashes:
            tier_num = self._infer_tier_from_test_id(test_id)
            if tier_num:
                config_pattern = f'config_tier{tier_num}_*.json'
                config_dir = self.workspace_root / 'config'
                if config_dir.exists():
                    matching_configs = list(config_dir.glob(config_pattern))
                    if matching_configs:
                        hashes['config_hash'] = self._compute_file_hash(matching_configs[0])

        # Source code hash (all Python files in src/)
        src_dir = self.workspace_root / 'src'
        if src_dir.exists():
            hashes['src_hash'] = self._compute_directory_hash(src_dir, ['*.py'])

        # Test file hash
        test_file = self._find_test_file(test_id)
        if test_file:
            hashes['test_hash'] = self._compute_file_hash(test_file)

        # Global settings hash
        settings_file = self.workspace_root / 'config' / 'settings.yaml'
        if settings_file.exists():
            hashes['settings_hash'] = self._compute_file_hash(settings_file)

        return hashes

    def _infer_tier_from_test_id(self, test_id: str) -> Optional[int]:
        """Infer tier number from test ID (e.g., EM-01 -> 5, GRAV-01 -> 2)"""
        prefix_to_tier = {
            'REL': 1,
            'GRAV': 2,
            'ENER': 3,
            'QUAN': 4,
            'EM': 5
        }

        for prefix, tier in prefix_to_tier.items():
            if test_id.startswith(prefix):
                return tier

        return None

    def _find_test_file(self, test_id: str) -> Optional[Path]:
        """Find test file for given test ID"""
        # Try common patterns
        patterns = [
            f'test_{test_id.lower()}.py',
            f'test_*{test_id.lower()}*.py'
        ]

        for pattern in patterns:
            matches = list(self.workspace_root.rglob(pattern))
            if matches:
                return matches[0]

        return None

    def is_cache_valid(self, test_id: str, config_file: Optional[Path] = None) -> bool:
        """
        Check if cached results are valid for this test.

        Returns True if:
        - Cache entry exists
        - All dependency hashes match current state
        """
        if test_id not in self.cache_index:
            return False

        cache_entry = self.cache_index[test_id]
        stored_hashes = cache_entry.get('dependency_hashes', {})

        # Compute current hashes
        current_hashes = self.compute_test_dependencies_hash(test_id, config_file)

        # Compare all hashes
        for key, current_hash in current_hashes.items():
            stored_hash = stored_hashes.get(key)
            if stored_hash != current_hash:
                return False

        # Check if cached results directory exists
        cache_dir = self.cache_root / test_id
        if not cache_dir.exists():
            return False

        return True

    def get_cached_results(self, test_id: str) -> Optional[Path]:
        """Get path to cached results directory, or None if invalid"""
        if not self.is_cache_valid(test_id):
            return None

        cache_dir = self.cache_root / test_id
        results_dir = cache_dir / 'results'

        if results_dir.exists():
            return results_dir

        return None

    def store_test_results(
        self,
        test_id: str,
        results_dir: Path,
        config_file: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store test results in cache.

        Args:
            test_id: Unique test identifier
            results_dir: Directory containing test results to cache
            config_file: Optional config file path
            metadata: Optional metadata dict (pass rate, duration, etc.)
        """
        if not results_dir.exists():
            return

        # Create cache directory
        cache_dir = self.cache_root / test_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Copy results
        cached_results = cache_dir / 'results'
        if cached_results.exists():
            shutil.rmtree(cached_results)
        shutil.copytree(results_dir, cached_results)

        # Compute and store dependency hashes
        dependency_hashes = self.compute_test_dependencies_hash(test_id, config_file)

        # Create manifest
        manifest = {
            'test_id': test_id,
            'cached_at': datetime.now().isoformat(),
            'dependency_hashes': dependency_hashes,
            'metadata': metadata or {}
        }

        manifest_path = cache_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        # Update cache index
        self.cache_index[test_id] = {
            'test_id': test_id,
            'cached_at': manifest['cached_at'],
            'dependency_hashes': dependency_hashes,
            'cache_dir': str(cache_dir)
        }

        self._save_cache_index()

    def invalidate_cache(self, test_id: str):
        """Invalidate cached results for a specific test"""
        if test_id in self.cache_index:
            cache_dir = Path(self.cache_index[test_id]['cache_dir'])
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            del self.cache_index[test_id]
            self._save_cache_index()

    def invalidate_all(self):
        """Invalidate entire cache"""
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)
        self.cache_index = {}
        self._save_cache_index()

    def clear_cache(self):
        """Alias for invalidate_all() - clear entire cache"""
        self.invalidate_all()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_cached_tests': len(self.cache_index),
            'cache_size_mb': self._get_cache_size_mb(),
            'cached_tests': list(self.cache_index.keys())
        }

    def _get_cache_size_mb(self) -> float:
        """Get total cache size in MB"""
        if not self.cache_root.exists():
            return 0.0

        total_size = 0
        for file_path in self.cache_root.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)
