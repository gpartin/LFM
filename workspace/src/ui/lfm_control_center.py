#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Control Center - Simple Console Interface
==============================================
A user-friendly menu system for running LFM tests and viewing results.
No web frameworks - just enhanced console interaction.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import json

# ANSI colors for better console output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m' 
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the LFM banner"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print("    LFM CONTROL CENTER - Lattice Field Medium")
    print("    Unified Physics Validation Suite")
    print("=" * 60)
    print(f"{Colors.END}")

def print_menu():
    """Print the main menu options"""
    print(f"\n{Colors.WHITE}{Colors.BOLD}MAIN MENU:{Colors.END}")
    print(f"{Colors.GREEN}1.{Colors.END} Run Fast Tests (4 tests, ~2 minutes)")
    print(f"{Colors.GREEN}2.{Colors.END} Run Single Tier")
    print(f"{Colors.GREEN}3.{Colors.END} Run Specific Test")
    print(f"{Colors.GREEN}4.{Colors.END} Run All Tiers (Full Suite)")
    print(f"{Colors.YELLOW}5.{Colors.END} View Test Results")
    print(f"{Colors.YELLOW}6.{Colors.END} Run Emergence Validation")
    print(f"{Colors.YELLOW}7.{Colors.END} Generate Reports")
    print(f"{Colors.BLUE}8.{Colors.END} System Status")
    print(f"{Colors.RED}9.{Colors.END} Exit")
    print()

def get_test_status():
    """Get summary of recent test results"""
    try:
        from utils.lfm_results import get_results_root
        master_status = get_results_root() / "MASTER_TEST_STATUS.csv"
        if master_status.exists():
            with open(master_status, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    total_tests = len(lines) - 1
                    passed_tests = sum(1 for line in lines[1:] if 'PASS' in line)
                    return f"{passed_tests}/{total_tests} tests passing"
    except:
        pass
    return "No test results found"

def run_command_with_progress(cmd: List[str], description: str):
    """Run a command and show progress"""
    print(f"\n{Colors.BLUE}Starting: {description}{Colors.END}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Show live output
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\n{Colors.GREEN}✓ Completed successfully in {elapsed:.1f}s{Colors.END}")
    else:
        print(f"\n{Colors.RED}✗ Failed with exit code {process.returncode}{Colors.END}")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
    return process.returncode == 0

def select_tier():
    """Let user select which tier to run"""
    print(f"\n{Colors.WHITE}Select Tier:{Colors.END}")
    print(f"{Colors.GREEN}1.{Colors.END} Tier 1 - Relativistic (15 tests)")
    print(f"{Colors.GREEN}2.{Colors.END} Tier 2 - Gravity Analogue (25 tests)")  
    print(f"{Colors.GREEN}3.{Colors.END} Tier 3 - Energy Conservation (11 tests)")
    print(f"{Colors.GREEN}4.{Colors.END} Tier 4 - Quantization (14 tests)")
    
    while True:
        choice = input(f"\n{Colors.CYAN}Enter tier number (1-4): {Colors.END}")
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print(f"{Colors.RED}Invalid choice. Please enter 1, 2, 3, or 4.{Colors.END}")

def select_specific_test():
    """Let user select a specific test to run"""
    print(f"\n{Colors.WHITE}Enter test ID (e.g., REL-01, GRAV-12, ENER-03, QUAN-05):{Colors.END}")
    test_id = input(f"{Colors.CYAN}Test ID: {Colors.END}").strip().upper()
    
    # Basic validation
    valid_prefixes = ['REL', 'GRAV', 'ENER', 'QUAN']
    if any(test_id.startswith(prefix) for prefix in valid_prefixes):
        return test_id
    else:
        print(f"{Colors.RED}Invalid test ID format. Use REL-xx, GRAV-xx, ENER-xx, or QUAN-xx{Colors.END}")
        return None

def view_results_menu():
    """Show results viewing options"""
    clear_screen()
    print_header()
    print(f"\n{Colors.WHITE}{Colors.BOLD}RESULTS VIEWER:{Colors.END}")
    
    # Check what results exist
    from utils.lfm_results import get_results_root
    results_dir = get_results_root()
    if not results_dir.exists():
        print(f"{Colors.RED}No results directory found.{Colors.END}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
        return
    
    categories = ["Relativistic", "Gravity", "Energy", "Quantization"]
    for i, category in enumerate(categories, 1):
        cat_dir = results_dir / category
        if cat_dir.exists():
            test_count = len([d for d in cat_dir.iterdir() if d.is_dir()])
            print(f"{Colors.GREEN}{i}.{Colors.END} {category} ({test_count} tests)")
        else:
            print(f"{Colors.YELLOW}{i}.{Colors.END} {category} (no results)")
    
    print(f"{Colors.BLUE}5.{Colors.END} Open results folder in explorer")
    print(f"{Colors.RED}6.{Colors.END} Back to main menu")
    
    choice = input(f"\n{Colors.CYAN}Select option: {Colors.END}")
    
    if choice == '5':
        # Open results folder
        if os.name == 'nt':  # Windows
            os.startfile(str(results_dir))
        else:  # Linux/Mac
            subprocess.run(['xdg-open', str(results_dir)])
    elif choice in ['1', '2', '3', '4']:
        category = categories[int(choice) - 1]
        cat_dir = results_dir / category
        if cat_dir.exists():
            if os.name == 'nt':
                os.startfile(str(cat_dir))
            else:
                subprocess.run(['xdg-open', str(cat_dir)])

def run_emergence_test():
    """Run the emergence validation test"""
    clear_screen()
    print_header()
    print(f"\n{Colors.WHITE}{Colors.BOLD}EMERGENCE VALIDATION TEST:{Colors.END}")
    print("This test demonstrates spontaneous χ-field structure formation")
    print("from uniform initial conditions - key evidence for genuine emergence.")
    print()
    
    confirm = input(f"{Colors.CYAN}Run emergence test? (y/N): {Colors.END}")
    if confirm.lower() == 'y':
        cmd = ["python", "docs/evidence/emergence_validation/test_emergence_proof.py"]
        return run_command_with_progress(cmd, "Emergence Validation Test")

def generate_reports():
    """Generate various reports"""
    clear_screen()
    print_header()
    print(f"\n{Colors.WHITE}{Colors.BOLD}REPORT GENERATION:{Colors.END}")
    print(f"{Colors.GREEN}1.{Colors.END} Update Master Test Status")
    print(f"{Colors.GREEN}2.{Colors.END} Generate Results Report")
    print(f"{Colors.GREEN}3.{Colors.END} Build Upload Package (Zenodo/OSF)")
    print(f"{Colors.RED}4.{Colors.END} Back to main menu")
    
    choice = input(f"\n{Colors.CYAN}Select option: {Colors.END}")
    
    if choice == '1':
        cmd = ["python", "-c", "from utils.lfm_results import update_master_test_status; update_master_test_status()"]
        return run_command_with_progress(cmd, "Updating Master Test Status")
    elif choice == '2':
        cmd = ["python", "tools/compile_results_report.py"]
        return run_command_with_progress(cmd, "Generating Results Report")
    elif choice == '3':
        cmd = ["python", "tools/build_upload_package.py"]
        return run_command_with_progress(cmd, "Building Upload Package")

def show_system_status():
    """Show system information and status"""
    clear_screen()
    print_header()
    print(f"\n{Colors.WHITE}{Colors.BOLD}SYSTEM STATUS:{Colors.END}")
    
    # Python version
    print(f"Python Version: {sys.version}")
    
    # GPU status
    try:
        import cupy
        print(f"{Colors.GREEN}✓ GPU (CuPy) Available{Colors.END}")
    except ImportError:
        print(f"{Colors.YELLOW}⚠ GPU (CuPy) Not Available - CPU Only{Colors.END}")
    
    # Test status
    status = get_test_status()
    print(f"Last Test Results: {status}")
    
    # Disk space in results
    from utils.lfm_results import get_results_root
    results_dir = get_results_root()
    if results_dir.exists():
        total_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"Results Directory Size: {size_mb:.1f} MB")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")

def main():
    """Main control loop"""
    while True:
        clear_screen()
        print_header()
        
        # Show quick status
        status = get_test_status()
        print(f"{Colors.WHITE}Status: {status}{Colors.END}")
        
        print_menu()
        
        choice = input(f"{Colors.CYAN}Select option (1-9): {Colors.END}")
        
        if choice == '1':
            # Fast tests
            cmd = ["python", "run_parallel_tests.py", "--fast"]
            run_command_with_progress(cmd, "Fast Test Suite (4 tests)")
            
        elif choice == '2':
            # Single tier
            tier = select_tier()
            if tier:
                cmd = ["python", "run_parallel_tests.py", "--tiers", str(tier)]
                run_command_with_progress(cmd, f"Tier {tier} Test Suite")
        
        elif choice == '3':
            # Specific test
            test_id = select_specific_test()
            if test_id:
                cmd = ["python", "run_parallel_tests.py", "--tests", test_id]
                run_command_with_progress(cmd, f"Test {test_id}")
        
        elif choice == '4':
            # All tiers
            print(f"\n{Colors.YELLOW}Warning: This will run all 55 tests and may take 30+ minutes.{Colors.END}")
            confirm = input(f"{Colors.CYAN}Continue? (y/N): {Colors.END}")
            if confirm.lower() == 'y':
                cmd = ["python", "run_parallel_tests.py", "--tiers", "1,2,3,4"]
                run_command_with_progress(cmd, "Full Test Suite (All Tiers)")
        
        elif choice == '5':
            # View results
            view_results_menu()
        
        elif choice == '6':
            # Emergence test
            run_emergence_test()
        
        elif choice == '7':
            # Generate reports
            generate_reports()
        
        elif choice == '8':
            # System status
            show_system_status()
        
        elif choice == '9':
            # Exit
            print(f"\n{Colors.GREEN}Thank you for using LFM Control Center!{Colors.END}")
            break
        
        else:
            print(f"{Colors.RED}Invalid choice. Please enter a number 1-9.{Colors.END}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Goodbye!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        input("Press Enter to exit...")