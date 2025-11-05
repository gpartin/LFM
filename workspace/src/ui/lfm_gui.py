#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Control Center - Simple Windows GUI
========================================
A basic Tkinter-based GUI for running LFM tests.
No external dependencies - uses Python's built-in GUI library.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import time
import queue
from pathlib import Path
import os

class LFMControlCenter:
    def __init__(self, root):
        self.root = root
        self.root.title("LFM Control Center - Lattice Field Medium")
        self.root.geometry("800x600")
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Current process
        self.current_process = None
        
        self.setup_ui()
        self.check_queue()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Test Execution
        self.test_frame = ttk.Frame(notebook)
        notebook.add(self.test_frame, text="Run Tests")
        self.setup_test_tab()
        
        # Tab 2: Results Viewer
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="View Results")
        self.setup_results_tab()
        
        # Tab 3: Tools
        self.tools_frame = ttk.Frame(notebook)
        notebook.add(self.tools_frame, text="Tools & Reports")
        self.setup_tools_tab()
        
        # Status bar at bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_test_tab(self):
        """Setup the test execution tab"""
        # Header
        header = ttk.Label(self.test_frame, text="LFM Test Suite Runner", 
                          font=("Arial", 14, "bold"))
        header.pack(pady=10)
        
        # Quick options frame
        quick_frame = ttk.LabelFrame(self.test_frame, text="Quick Actions")
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(quick_frame, text="Fast Tests (4 tests, ~2 min)", 
                  command=self.run_fast_tests).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(quick_frame, text="Emergence Test", 
                  command=self.run_emergence_test).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Tier selection frame
        tier_frame = ttk.LabelFrame(self.test_frame, text="Run by Tier")
        tier_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Tier checkboxes
        self.tier_vars = {}
        tier_info = [
            (1, "Relativistic (15 tests)"),
            (2, "Gravity Analogue (25 tests)"), 
            (3, "Energy Conservation (11 tests)"),
            (4, "Quantization (14 tests)")
        ]
        
        for tier, desc in tier_info:
            var = tk.BooleanVar()
            self.tier_vars[tier] = var
            ttk.Checkbutton(tier_frame, text=f"Tier {tier}: {desc}", 
                           variable=var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(tier_frame, text="Run Selected Tiers", 
                  command=self.run_selected_tiers).pack(pady=5)
        
        # Specific test frame
        specific_frame = ttk.LabelFrame(self.test_frame, text="Run Specific Test")
        specific_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(specific_frame, text="Test ID:").pack(side=tk.LEFT, padx=5)
        self.test_id_var = tk.StringVar()
        test_entry = ttk.Entry(specific_frame, textvariable=self.test_id_var, width=15)
        test_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(specific_frame, text="(e.g., REL-01, GRAV-12)").pack(side=tk.LEFT, padx=5)
        ttk.Button(specific_frame, text="Run Test", 
                  command=self.run_specific_test).pack(side=tk.RIGHT, padx=5)
        
        # Output area
        output_frame = ttk.LabelFrame(self.test_frame, text="Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(output_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Clear Output", 
                  command=self.clear_output).pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=5)
    
    def setup_results_tab(self):
        """Setup the results viewing tab"""
        ttk.Label(self.results_frame, text="Test Results Browser", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Results tree
        tree_frame = ttk.Frame(self.results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_tree = ttk.Treeview(tree_frame, columns=("status", "runtime"), height=15)
        self.results_tree.heading("#0", text="Test")
        self.results_tree.heading("status", text="Status")
        self.results_tree.heading("runtime", text="Runtime")
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh Results", 
                  command=self.refresh_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open Results Folder", 
                  command=self.open_results_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Test Details", 
                  command=self.view_test_details).pack(side=tk.RIGHT, padx=5)
        
        # Load initial results
        self.refresh_results()
    
    def setup_tools_tab(self):
        """Setup the tools and reports tab"""
        ttk.Label(self.tools_frame, text="Tools & Report Generation", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Report generation
        report_frame = ttk.LabelFrame(self.tools_frame, text="Report Generation")
        report_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(report_frame, text="Update Master Test Status", 
                  command=self.update_test_status).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(report_frame, text="Generate Results Report", 
                  command=self.generate_results_report).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(report_frame, text="Build Upload Package", 
                  command=self.build_upload_package).pack(fill=tk.X, padx=5, pady=2)
        
        # System info
        info_frame = ttk.LabelFrame(self.tools_frame, text="System Information")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(info_frame, text="Refresh System Info", 
                  command=self.refresh_system_info).pack(pady=5)
        
        # Load initial system info
        self.refresh_system_info()
    
    def run_fast_tests(self):
        """Run the fast test suite"""
        cmd = ["python", "run_parallel_tests.py", "--fast"]
        self.run_command(cmd, "Running fast tests...")
    
    def run_emergence_test(self):
        """Run the emergence validation test"""
        cmd = ["python", "docs/evidence/emergence_validation/test_emergence_proof.py"]
        self.run_command(cmd, "Running emergence validation test...")
    
    def run_selected_tiers(self):
        """Run tests from selected tiers"""
        selected_tiers = [str(tier) for tier, var in self.tier_vars.items() if var.get()]
        
        if not selected_tiers:
            messagebox.showwarning("No Selection", "Please select at least one tier.")
            return
        
        cmd = ["python", "run_parallel_tests.py", "--tiers", ",".join(selected_tiers)]
        self.run_command(cmd, f"Running tier(s) {', '.join(selected_tiers)}...")
    
    def run_specific_test(self):
        """Run a specific test"""
        test_id = self.test_id_var.get().strip().upper()
        if not test_id:
            messagebox.showwarning("No Test ID", "Please enter a test ID.")
            return
        
        cmd = ["python", "run_parallel_tests.py", "--tests", test_id]
        self.run_command(cmd, f"Running test {test_id}...")
    
    def run_command(self, cmd, status_msg):
        """Run a command in a separate thread"""
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Process Running", "A test is already running. Please wait or stop it first.")
            return
        
        self.status_var.set(status_msg)
        self.stop_button.config(state=tk.NORMAL)
        self.clear_output()
        
        def run_in_thread():
            try:
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                for line in self.current_process.stdout:
                    self.output_queue.put(("output", line.rstrip()))
                
                self.current_process.wait()
                self.output_queue.put(("done", self.current_process.returncode))
                
            except Exception as e:
                self.output_queue.put(("error", str(e)))
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    
    def stop_process(self):
        """Stop the current process"""
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.output_queue.put(("output", "\n--- Process terminated by user ---"))
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Process stopped")
    
    def check_queue(self):
        """Check for messages from background threads"""
        try:
            while True:
                msg_type, msg_data = self.output_queue.get_nowait()
                
                if msg_type == "output":
                    self.output_text.insert(tk.END, msg_data + "\n")
                    self.output_text.see(tk.END)
                elif msg_type == "done":
                    self.stop_button.config(state=tk.DISABLED)
                    if msg_data == 0:
                        self.status_var.set("Completed successfully")
                    else:
                        self.status_var.set(f"Failed with exit code {msg_data}")
                    self.refresh_results()  # Update results after test completion
                elif msg_type == "error":
                    self.output_text.insert(tk.END, f"Error: {msg_data}\n")
                    self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Error occurred")
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete(1.0, tk.END)
    
    def refresh_results(self):
        """Refresh the results tree"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        results_dir = Path("results")
        if not results_dir.exists():
            return
        
        categories = ["Relativistic", "Gravity", "Energy", "Quantization"]
        for category in categories:
            cat_dir = results_dir / category
            if cat_dir.exists():
                # Add category node
                cat_node = self.results_tree.insert("", tk.END, text=category, values=("", ""))
                
                # Add test results
                for test_dir in sorted(cat_dir.iterdir()):
                    if test_dir.is_dir():
                        # Try to read summary.json
                        summary_file = test_dir / "summary.json"
                        status = "Unknown"
                        runtime = ""
                        
                        if summary_file.exists():
                            try:
                                import json
                                with open(summary_file) as f:
                                    data = json.load(f)
                                    status = "PASS" if data.get("passed", False) else "FAIL"
                                    runtime = f"{data.get('runtime_sec', 0):.1f}s"
                            except:
                                pass
                        
                        self.results_tree.insert(cat_node, tk.END, text=test_dir.name, 
                                               values=(status, runtime))
        
        # Expand all categories
        for item in self.results_tree.get_children():
            self.results_tree.item(item, open=True)
    
    def open_results_folder(self):
        """Open the results folder in file explorer"""
        results_dir = Path("results")
        if results_dir.exists():
            if os.name == 'nt':  # Windows
                os.startfile(str(results_dir))
            else:  # Linux/Mac
                subprocess.run(['xdg-open', str(results_dir)])
    
    def view_test_details(self):
        """View details of selected test"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a test to view details.")
            return
        
        item = selection[0]
        parent = self.results_tree.parent(item)
        
        if not parent:  # Category selected, not test
            messagebox.showinfo("Invalid Selection", "Please select a specific test, not a category.")
            return
        
        test_name = self.results_tree.item(item)["text"]
        category = self.results_tree.item(parent)["text"]
        
        test_dir = Path("results") / category / test_name
        if test_dir.exists():
            if os.name == 'nt':
                os.startfile(str(test_dir))
            else:
                subprocess.run(['xdg-open', str(test_dir)])
    
    def update_test_status(self):
        """Update master test status"""
        cmd = ["python", "-c", "from utils.lfm_results import update_master_test_status; update_master_test_status()"]
        self.run_command(cmd, "Updating master test status...")
    
    def generate_results_report(self):
        """Generate results report"""
        cmd = ["python", "tools/compile_results_report.py"]
        self.run_command(cmd, "Generating results report...")
    
    def build_upload_package(self):
        """Build upload package"""
        cmd = ["python", "tools/build_upload_package.py"]
        self.run_command(cmd, "Building upload package...")
    
    def refresh_system_info(self):
        """Refresh system information display"""
        import sys
        import platform
        
        info = []
        info.append(f"Python Version: {sys.version}")
        info.append(f"Platform: {platform.platform()}")
        info.append(f"Architecture: {platform.architecture()[0]}")
        
        # Check GPU availability
        try:
            import cupy
            info.append("GPU (CuPy): Available ✓")
        except ImportError:
            info.append("GPU (CuPy): Not Available ⚠")
        
        # Check test results
        try:
            master_status = Path("results/MASTER_TEST_STATUS.csv")
            if master_status.exists():
                with open(master_status, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        total_tests = len(lines) - 1
                        passed_tests = sum(1 for line in lines[1:] if 'PASS' in line)
                        info.append(f"Test Results: {passed_tests}/{total_tests} passing")
            else:
                info.append("Test Results: No results found")
        except:
            info.append("Test Results: Error reading status")
        
        # Update display
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info))
        self.info_text.config(state=tk.DISABLED)

def main():
    """Main entry point"""
    root = tk.Tk()
    app = LFMControlCenter(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()