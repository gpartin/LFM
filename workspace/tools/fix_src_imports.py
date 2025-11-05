#!/usr/bin/env python3
"""
Fix imports after src/ reorganization.

Updates imports in workspace/src/ files to use new subdirectory structure:
- lfm_equation, lfm_simulator, lfm_backend, lfm_parallel, lfm_fields -> core.*
- lfm_gui, lfm_console, lfm_control_center, lfm_studio_ide, lfm_visualizer, lfm_plotting -> ui.*
- lfm_config, lfm_logger, lfm_diagnostics, lfm_results, path_utils, numeric_integrity, resource_*, energy_monitor -> utils.*
- lfm_test_harness, lfm_test_metrics, lfm_tiers, StandardTierTemplate -> harness.*
- chi_field_equation, lorentz_transform, em_analytical_framework, em_test_implementation_template -> physics.*
"""

import re
from pathlib import Path

# Mapping of modules to their new locations
MODULE_MAPPING = {
    # Core modules
    'lfm_equation': 'core.lfm_equation',
    'lfm_simulator': 'core.lfm_simulator',
    'lfm_backend': 'core.lfm_backend',
    'lfm_parallel': 'core.lfm_parallel',
    'lfm_fields': 'core.lfm_fields',
    
    # UI modules
    'lfm_gui': 'ui.lfm_gui',
    'lfm_console': 'ui.lfm_console',
    'lfm_control_center': 'ui.lfm_control_center',
    'lfm_studio_ide': 'ui.lfm_studio_ide',
    'lfm_visualizer': 'ui.lfm_visualizer',
    'lfm_plotting': 'ui.lfm_plotting',
    
    # Utils modules
    'lfm_config': 'utils.lfm_config',
    'lfm_logger': 'utils.lfm_logger',
    'lfm_diagnostics': 'utils.lfm_diagnostics',
    'lfm_results': 'utils.lfm_results',
    'path_utils': 'utils.path_utils',
    'numeric_integrity': 'utils.numeric_integrity',
    'resource_monitor': 'utils.resource_monitor',
    'resource_tracking': 'utils.resource_tracking',
    'energy_monitor': 'utils.energy_monitor',
    
    # Harness modules
    'lfm_test_harness': 'harness.lfm_test_harness',
    'lfm_test_metrics': 'harness.lfm_test_metrics',
    'lfm_tiers': 'harness.lfm_tiers',
    'StandardTierTemplate': 'harness.StandardTierTemplate',
    
    # Physics modules
    'chi_field_equation': 'physics.chi_field_equation',
    'lorentz_transform': 'physics.lorentz_transform',
    'em_analytical_framework': 'physics.em_analytical_framework',
    'em_test_implementation_template': 'physics.em_test_implementation_template',
}


def fix_imports(file_path: Path) -> int:
    """Fix imports in a single file. Returns number of changes made."""
    content = file_path.read_text(encoding='utf-8')
    original = content
    changes = 0
    
    for old_module, new_module in MODULE_MAPPING.items():
        # Pattern 1: from module import ...
        pattern1 = rf'from {re.escape(old_module)} import'
        replacement1 = f'from {new_module} import'
        content, n1 = re.subn(pattern1, replacement1, content)
        changes += n1
        
        # Pattern 2: import module
        pattern2 = rf'\bimport {re.escape(old_module)}\b'
        replacement2 = f'import {new_module}'
        content, n2 = re.subn(pattern2, replacement2, content)
        changes += n2
    
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"✓ Fixed {changes} imports in {file_path.relative_to(ROOT)}")
    
    return changes


def main():
    global ROOT
    ROOT = Path(__file__).resolve().parent.parent
    src_dir = ROOT / 'workspace' / 'src'
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return 1
    
    total_changes = 0
    files_changed = 0
    
    # Process all Python files in src/
    for py_file in src_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        changes = fix_imports(py_file)
        if changes > 0:
            total_changes += changes
            files_changed += 1
    
    print()
    print(f"Summary: Fixed {total_changes} imports across {files_changed} files")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
