"""
Interactive Lattice Playground
Click to add energy, watch waves propagate in real-time.
Uses pygame for fast interactive visualization.
Press '3' to toggle 3D view.
"""

import numpy as np
import pygame
import sys
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for separate window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from lfm_equation import lattice_step, energy_total

# Physics parameters
GRID_SIZE = 128  # Lattice size (NxN)
DX = 0.5
DT = 0.1
CHI = 1.0  # Mass parameter
GAMMA = 0.0  # Damping (0 = conservative)
BOUNDARY = "periodic"
STENCIL_ORDER = 2

# Visualization parameters
WINDOW_SIZE = 800
SCALE_FACTOR = 5.0  # Color intensity scaling
FPS_TARGET = 60
STEPS_PER_FRAME = 1  # Physics steps per render frame

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (100, 100, 255)

class InteractiveLattice:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE + 200, WINDOW_SIZE))
        pygame.display.set_caption("LFM Interactive Lattice")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Initialize lattice state
        self.E = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.E_prev = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        
        # Physics parameters (mutable)
        self.chi = CHI
        self.gamma_damp = GAMMA
        self.dt = DT
        self.dx = DX
        
        # State
        self.paused = False
        self.step_count = 0
        self.energy_history = []
        self.max_history = 500
        
        # Interaction
        self.pulse_amplitude = 1.0
        self.pulse_width = 3.0
        
        # 3D visualization
        self.show_3d = True  # Default to 3D mode
        self.fig_3d = None
        self.ax_3d = None
        self.plot_3d_surf = None
        
        # Initialize 3D view on startup
        self.toggle_3d_view()  # This will create the 3D window
        
        # Scenario playback system
        self.scenario_mode = False
        self.scenario_frames = []  # List of (E, E_prev) states
        self.scenario_frame_index = 0
        self.scenario_playing = False
        self.scenario_name = ""
        self.playback_speed = 1  # Frames to advance per render (1 = slow, 5 = fast)
        
    def add_gaussian_pulse(self, grid_x, grid_y, amplitude=None):
        """Add a Gaussian pulse at grid coordinates."""
        if amplitude is None:
            amplitude = self.pulse_amplitude
            
        y_grid, x_grid = np.ogrid[0:GRID_SIZE, 0:GRID_SIZE]
        r_sq = (x_grid - grid_x)**2 + (y_grid - grid_y)**2
        pulse = amplitude * np.exp(-r_sq / (2 * self.pulse_width**2))
        self.E += pulse
        
    def add_line_pulse(self, grid_x, amplitude=None):
        """Add a vertical line pulse (plane wave)."""
        if amplitude is None:
            amplitude = self.pulse_amplitude
            
        x_grid = np.arange(GRID_SIZE)
        pulse_1d = amplitude * np.exp(-(x_grid - grid_x)**2 / (2 * self.pulse_width**2))
        self.E += pulse_1d[np.newaxis, :]
        
    def step_physics(self):
        """Advance physics by one timestep."""
        # Build params dict for lattice_step
        params = {
            'dt': self.dt,
            'dx': self.dx,
            'chi': self.chi,
            'gamma_damp': self.gamma_damp,
            'boundary': BOUNDARY,
            'stencil_order': STENCIL_ORDER,
            'alpha': 1.0,  # Wave equation coefficient (c^2 = alpha/beta)
            'beta': 1.0,   # Normalization (c = 1 for alpha=beta=1)
        }
        
        E_next = lattice_step(self.E, self.E_prev, params)
        self.E_prev = self.E
        self.E = E_next
        self.step_count += 1
        
        # Track energy
        c = 1.0  # Speed of light (from alpha=beta=1)
        energy = energy_total(self.E, self.E_prev, self.dt, self.dx, c, self.chi)
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
            
    def reset(self):
        """Reset lattice to zero state."""
        self.E = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.E_prev = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.step_count = 0
        self.energy_history = []
        self.exit_scenario_mode()
        
    def render_field(self):
        """Render the field E as a colored surface."""
        # Normalize field for visualization
        E_vis = self.E * SCALE_FACTOR
        E_vis = np.clip(E_vis, -1.0, 1.0)
        
        # Create RGB image (red = positive, blue = negative)
        rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        
        # Positive values -> red
        pos_mask = E_vis > 0
        rgb[pos_mask, 0] = (E_vis[pos_mask] * 255).astype(np.uint8)
        
        # Negative values -> blue
        neg_mask = E_vis < 0
        rgb[neg_mask, 2] = (-E_vis[neg_mask] * 255).astype(np.uint8)
        
        # Scale up to window size
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (WINDOW_SIZE, WINDOW_SIZE))
        self.screen.blit(surf, (0, 0))
        
        # Draw apparatus overlay for double slit scenario
        if self.scenario_mode and self.scenario_name == "Double Slit Experiment":
            self.render_double_slit_apparatus()
        
    def render_energy_graph(self):
        """Render energy history as a line graph."""
        if len(self.energy_history) < 2:
            return
            
        graph_x = WINDOW_SIZE + 10
        graph_y = 300
        graph_w = 180
        graph_h = 150
        
        # Draw background
        pygame.draw.rect(self.screen, (30, 30, 30), (graph_x, graph_y, graph_w, graph_h))
        pygame.draw.rect(self.screen, WHITE, (graph_x, graph_y, graph_w, graph_h), 1)
        
        # Normalize energy to graph height
        energies = np.array(self.energy_history)
        if energies.max() > 0:
            energies_norm = (energies / energies.max()) * (graph_h - 10)
        else:
            energies_norm = energies * 0
            
        # Draw line
        points = []
        for i, e_norm in enumerate(energies_norm):
            x = graph_x + 5 + int((i / len(energies_norm)) * (graph_w - 10))
            y = graph_y + graph_h - 5 - int(e_norm)
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, GREEN, False, points, 2)
            
        # Label
        label = self.small_font.render("Energy", True, WHITE)
        self.screen.blit(label, (graph_x + 5, graph_y + 5))
        
    def toggle_3d_view(self):
        """Toggle 3D matplotlib view on/off."""
        self.show_3d = not self.show_3d
        
        if self.show_3d and self.fig_3d is None:
            # Create 3D figure
            plt.ion()  # Interactive mode
            self.fig_3d = plt.figure(figsize=(10, 8))
            self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('Field E')
            self.ax_3d.set_title('LFM Lattice - 3D View')
            
        elif not self.show_3d and self.fig_3d is not None:
            plt.close(self.fig_3d)
            self.fig_3d = None
            self.ax_3d = None
            self.plot_3d_surf = None
            
    def update_3d_view(self):
        """Update the 3D matplotlib plot."""
        if not self.show_3d or self.ax_3d is None:
            return
            
        # Clear and redraw
        self.ax_3d.clear()
        
        # Create meshgrid
        x = np.arange(0, GRID_SIZE)
        y = np.arange(0, GRID_SIZE)
        X, Y = np.meshgrid(x, y)
        
        # Plot surface
        # Downsample for performance if grid is large
        stride = max(1, GRID_SIZE // 64)
        surf = self.ax_3d.plot_surface(
            X[::stride, ::stride], 
            Y[::stride, ::stride], 
            self.E[::stride, ::stride],
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        
        # Add apparatus visualization for double slit scenario
        if self.scenario_mode and self.scenario_name == "Double Slit Experiment":
            self.render_double_slit_apparatus_3d()
        
        # Set axis labels and limits
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Field E')
        
        # Add scenario info to title
        if self.scenario_mode:
            title = f'{self.scenario_name} - Frame {self.scenario_frame_index}/{len(self.scenario_frames)}'
        else:
            title = f'LFM Lattice 3D - Step {self.step_count}'
        self.ax_3d.set_title(title)
        
        # Auto-scale z axis to data
        z_max = max(abs(self.E.min()), abs(self.E.max()), 0.1)
        self.ax_3d.set_zlim(-z_max, z_max)
        
        # Add colorbar on first draw
        if self.plot_3d_surf is None:
            self.fig_3d.colorbar(surf, ax=self.ax_3d, shrink=0.5, aspect=5)
        self.plot_3d_surf = surf
        
        # Draw
        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()
        
    # ----------------------------------------------------------------
    # Scenario System
    # ----------------------------------------------------------------
    
    def load_double_slit_scenario(self):
        """
        Load and run the Double Slit Experiment scenario.
        
        Creates a plane wave from the left that encounters a barrier with
        two narrow slits, demonstrating wave interference patterns.
        """
        print("\n" + "="*60)
        print("Loading: THE DOUBLE SLIT EXPERIMENT")
        print("="*60)
        print("A plane wave encounters a barrier with two slits.")
        print("Watch the interference pattern form on the right side!")
        print("="*60 + "\n")
        
        # Reset to clean state
        self.E = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.E_prev = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        self.step_count = 0
        self.energy_history = []
        
        # Create barrier with two slits
        # Barrier at x = GRID_SIZE // 3
        barrier_x = GRID_SIZE // 3
        slit_width = 4  # Width of each slit
        slit_separation = 24  # Distance between slit centers
        center_y = GRID_SIZE // 2
        
        slit1_y = center_y - slit_separation // 2
        slit2_y = center_y + slit_separation // 2
        
        # Create plane wave (validated parameters from test)
        amplitude = 1.0
        wavelength = 10.0  # Spatial wavelength in grid units
        k_x = 2.0 * np.pi / wavelength  # wave number
        
        y_grid, x_grid = np.ogrid[0:GRID_SIZE, 0:GRID_SIZE]
        
        # Plane wave: E = A * sin(k*x)
        # For traveling wave, E_prev should be phase-shifted
        self.E[:, :10] = amplitude * np.sin(k_x * x_grid[:, :10])
        self.E_prev[:, :10] = amplitude * np.sin(k_x * (x_grid[:, :10] - self.dx))  # Shifted back for rightward motion
        
        # Store scenario metadata
        self.scenario_name = "Double Slit Experiment"
        self.scenario_frames = []
        self.scenario_frame_index = 0
        self.scenario_mode = True
        self.scenario_playing = True
        
        # Pre-compute frames for smooth playback
        print("Computing simulation frames...")
        print(f"  Barrier at x={barrier_x}, slits at y={slit1_y} and y={slit2_y}")
        
        # Save original parameters
        temp_chi = self.chi
        temp_gamma = self.gamma_damp
        
        # Use validated parameters from test
        self.chi = 0.0
        self.gamma_damp = 0.01  # Low damping to preserve wave
        
        # Run simulation
        max_frames = 800  # Enough to see pattern develop
        save_every = 2  # Save every 2nd frame for 400 frames
        source_duration = 300  # Pump waves for longer
        
        # Compute wave parameters for dispersion relation
        # Using alpha=1.0, beta=1.0 (standard wave equation)
        omega = k_x * np.sqrt(1.0 / 1.0)  # = k_x
        
        for i in range(max_frames):
            # Add continuous plane wave source (left side)
            if i < source_duration:
                source_wave = amplitude * np.sin(omega * i * self.dt)
                self.E[:, 8] = source_wave  # Set entire column to plane wave value
            
            # Barrier: zero field except at slits
            if i > 15:  # Start blocking after initial wave forms
                for y in range(GRID_SIZE):
                    # Check if this row is in a slit
                    in_slit = False
                    if (slit1_y <= y < slit1_y + slit_width) or (slit2_y <= y < slit2_y + slit_width):
                        in_slit = True
                    
                    if not in_slit:
                        self.E[y, barrier_x] = 0.0
                        self.E_prev[y, barrier_x] = 0.0
                        
            self.step_physics()
            
            # Save frame
            if i % save_every == 0:
                self.scenario_frames.append((self.E.copy(), self.E_prev.copy()))
                
            # Progress indicator with diagnostics
            if i % 100 == 0:
                max_field = np.max(np.abs(self.E))
                field_at_barrier = np.mean(np.abs(self.E[:, barrier_x]))
                screen_x = int(GRID_SIZE * 0.75)
                field_at_screen = np.mean(np.abs(self.E[:, screen_x]))
                print(f"  Frame {i}/{max_frames}: max={max_field:.3f}, "
                      f"barrier={field_at_barrier:.3f}, screen={field_at_screen:.3f}")
                      
        # Restore original parameters
        self.chi = temp_chi
        self.gamma_damp = temp_gamma
        print(f"Scenario loaded! {len(self.scenario_frames)} frames ready.")
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT: Step backward/forward")
        print("  [/]: Decrease/Increase playback speed")
        print("  ESC: Exit scenario mode")
        print("  3: Toggle 3D view")
        
        # Start at beginning
        self.scenario_frame_index = 0
        self.load_scenario_frame(0)
        
    def load_scenario_frame(self, index):
        """Load a specific frame from the scenario."""
        if 0 <= index < len(self.scenario_frames):
            self.scenario_frame_index = index
            self.E, self.E_prev = self.scenario_frames[index]
            self.E = self.E.copy()
            self.E_prev = self.E_prev.copy()
            
    def exit_scenario_mode(self):
        """Exit scenario playback and return to interactive mode."""
        self.scenario_mode = False
        self.scenario_frames = []
        self.scenario_frame_index = 0
        self.scenario_playing = False
        self.scenario_name = ""
        print("Exited scenario mode - back to interactive mode")
        
    def render_double_slit_apparatus(self):
        """
        Render the physical apparatus overlay for double slit experiment.
        Shows the barrier with slits and the detection screen.
        """
        # Calculate positions (same as in scenario setup)
        barrier_x = GRID_SIZE // 3
        slit_width = 4
        slit_separation = 24
        center_y = GRID_SIZE // 2
        slit1_y = center_y - slit_separation // 2
        slit2_y = center_y + slit_separation // 2
        
        # Convert grid coords to screen coords
        def grid_to_screen(gx, gy):
            sx = int((gx / GRID_SIZE) * WINDOW_SIZE)
            sy = int((gy / GRID_SIZE) * WINDOW_SIZE)
            return (sx, sy)
        
        # Draw barrier (dark gray wall with slits)
        barrier_screen_x = int((barrier_x / GRID_SIZE) * WINDOW_SIZE)
        barrier_thickness = 8
        barrier_color = (60, 60, 60)  # Dark gray
        
        # Top section of barrier (above slit 1)
        pygame.draw.rect(self.screen, barrier_color,
                        (barrier_screen_x - barrier_thickness//2, 0,
                         barrier_thickness, int((slit1_y - slit_width) / GRID_SIZE * WINDOW_SIZE)))
        
        # Middle section (between slits)
        middle_start_y = int((slit1_y + slit_width) / GRID_SIZE * WINDOW_SIZE)
        middle_end_y = int((slit2_y - slit_width) / GRID_SIZE * WINDOW_SIZE)
        pygame.draw.rect(self.screen, barrier_color,
                        (barrier_screen_x - barrier_thickness//2, middle_start_y,
                         barrier_thickness, middle_end_y - middle_start_y))
        
        # Bottom section (below slit 2)
        bottom_start_y = int((slit2_y + slit_width) / GRID_SIZE * WINDOW_SIZE)
        pygame.draw.rect(self.screen, barrier_color,
                        (barrier_screen_x - barrier_thickness//2, bottom_start_y,
                         barrier_thickness, WINDOW_SIZE - bottom_start_y))
        
        # Draw slit openings (bright yellow outline)
        slit_color = (255, 255, 100)
        slit1_screen_y = int((slit1_y) / GRID_SIZE * WINDOW_SIZE)
        slit2_screen_y = int((slit2_y) / GRID_SIZE * WINDOW_SIZE)
        slit_height = int((slit_width * 2) / GRID_SIZE * WINDOW_SIZE)
        
        pygame.draw.rect(self.screen, slit_color,
                        (barrier_screen_x - barrier_thickness//2 - 2, slit1_screen_y - slit_height//2,
                         barrier_thickness + 4, slit_height), 2)
        pygame.draw.rect(self.screen, slit_color,
                        (barrier_screen_x - barrier_thickness//2 - 2, slit2_screen_y - slit_height//2,
                         barrier_thickness + 4, slit_height), 2)
        
        # Draw detection screen on the right (semi-transparent white)
        screen_x = int((GRID_SIZE * 0.75) / GRID_SIZE * WINDOW_SIZE)
        screen_color = (200, 200, 200, 100)
        screen_surf = pygame.Surface((3, WINDOW_SIZE), pygame.SRCALPHA)
        screen_surf.fill(screen_color)
        self.screen.blit(screen_surf, (screen_x, 0))
        
        # Draw intensity pattern on detection screen (integrated field strength)
        # Sample the field at the screen location and draw dots
        screen_grid_x = int(GRID_SIZE * 0.75)
        if screen_grid_x < GRID_SIZE:
            for y in range(0, GRID_SIZE, 2):  # Sample every 2 pixels
                intensity = abs(self.E[y, screen_grid_x])
                if intensity > 0.01:  # Threshold for visibility
                    # Map intensity to brightness (yellow dots)
                    brightness = min(255, int(intensity * 500))
                    dot_color = (brightness, brightness, 0)
                    screen_y = int((y / GRID_SIZE) * WINDOW_SIZE)
                    pygame.draw.circle(self.screen, dot_color, (screen_x + 10, screen_y), 2)
        
        # Draw labels
        label_font = pygame.font.Font(None, 20)
        
        # Source label (left side)
        source_label = label_font.render("Wave Source", True, (255, 255, 255))
        self.screen.blit(source_label, (10, 10))
        
        # Barrier label
        barrier_label = label_font.render("Barrier", True, (255, 255, 100))
        self.screen.blit(barrier_label, (barrier_screen_x - 30, WINDOW_SIZE - 30))
        
        # Screen label
        screen_label = label_font.render("Detection Screen", True, (200, 200, 200))
        self.screen.blit(screen_label, (screen_x + 15, 10))
        
        # Draw "Interference Pattern" label with arrow
        if self.scenario_frame_index > 200:  # Show after pattern develops
            pattern_label = label_font.render("Interference Pattern", True, (255, 255, 0))
            self.screen.blit(pattern_label, (screen_x + 15, WINDOW_SIZE // 2 - 20))
            # Draw arrow pointing to fringes
            arrow_start = (screen_x + 15, WINDOW_SIZE // 2 - 10)
            arrow_end = (screen_x + 5, WINDOW_SIZE // 2)
            pygame.draw.line(self.screen, (255, 255, 0), arrow_start, arrow_end, 2)
            pygame.draw.polygon(self.screen, (255, 255, 0), [
                arrow_end,
                (arrow_end[0] + 5, arrow_end[1] - 3),
                (arrow_end[0] + 5, arrow_end[1] + 3)
            ])
            
    def render_double_slit_apparatus_3d(self):
        """
        Render the physical apparatus in the 3D plot.
        Shows barrier, slits, and detection screen as 3D objects.
        """
        # Calculate positions (same as in scenario setup)
        barrier_x = GRID_SIZE // 3
        slit_width = 4
        slit_separation = 24
        center_y = GRID_SIZE // 2
        slit1_y = center_y - slit_separation // 2
        slit2_y = center_y + slit_separation // 2
        
        z_max = max(abs(self.E.min()), abs(self.E.max()), 0.1)
        barrier_height = z_max * 1.5  # Tall enough to be visible
        
        # Draw barrier as vertical planes (gray)
        # Top section (above slit 1)
        y_top = np.array([0, slit1_y - slit_width])
        x_barrier = np.array([barrier_x, barrier_x])
        X_top, Y_top = np.meshgrid(x_barrier, y_top)
        Z_top = np.ones_like(X_top) * barrier_height
        self.ax_3d.plot_surface(X_top, Y_top, Z_top, color='gray', alpha=0.7)
        
        # Also draw at bottom (make it a wall)
        Z_bottom = np.ones_like(X_top) * (-barrier_height)
        for z in np.linspace(-barrier_height, barrier_height, 3):
            self.ax_3d.plot_surface(X_top, Y_top, np.ones_like(X_top) * z, 
                                   color='gray', alpha=0.3)
        
        # Middle section (between slits)
        y_mid = np.array([slit1_y + slit_width, slit2_y - slit_width])
        X_mid, Y_mid = np.meshgrid(x_barrier, y_mid)
        Z_mid = np.ones_like(X_mid) * barrier_height
        self.ax_3d.plot_surface(X_mid, Y_mid, Z_mid, color='gray', alpha=0.7)
        for z in np.linspace(-barrier_height, barrier_height, 3):
            self.ax_3d.plot_surface(X_mid, Y_mid, np.ones_like(X_mid) * z,
                                   color='gray', alpha=0.3)
        
        # Bottom section (below slit 2)
        y_bot = np.array([slit2_y + slit_width, GRID_SIZE])
        X_bot, Y_bot = np.meshgrid(x_barrier, y_bot)
        Z_bot = np.ones_like(X_bot) * barrier_height
        self.ax_3d.plot_surface(X_bot, Y_bot, Z_bot, color='gray', alpha=0.7)
        for z in np.linspace(-barrier_height, barrier_height, 3):
            self.ax_3d.plot_surface(X_bot, Y_bot, np.ones_like(X_bot) * z,
                                   color='gray', alpha=0.3)
        
        # Draw slit markers (yellow lines)
        for slit_y in [slit1_y, slit2_y]:
            self.ax_3d.plot([barrier_x, barrier_x], 
                           [slit_y - slit_width, slit_y + slit_width],
                           [barrier_height, barrier_height],
                           'y-', linewidth=3, label='Slit' if slit_y == slit1_y else '')
        
        # Draw detection screen (semi-transparent vertical plane)
        screen_x = int(GRID_SIZE * 0.75)
        y_screen = np.array([0, GRID_SIZE])
        x_screen = np.array([screen_x, screen_x])
        X_screen, Y_screen = np.meshgrid(x_screen, y_screen)
        Z_screen = np.ones_like(X_screen) * (barrier_height * 0.5)
        self.ax_3d.plot_surface(X_screen, Y_screen, Z_screen, 
                               color='lightgray', alpha=0.2)
        
        # Add text labels
        self.ax_3d.text(GRID_SIZE // 6, GRID_SIZE // 2, barrier_height, 
                       'Wave\nSource', fontsize=10, color='white')
        self.ax_3d.text(barrier_x, GRID_SIZE - 10, barrier_height, 
                       'Barrier\n(2 slits)', fontsize=10, color='yellow')
        self.ax_3d.text(screen_x + 5, GRID_SIZE // 2, barrier_height * 0.5, 
                       'Detection\nScreen', fontsize=10, color='lightgray')
        
    def render_ui(self):
        """Render UI overlay with stats and controls."""
        panel_x = WINDOW_SIZE + 10
        y_offset = 20
        
        # Title
        title = self.font.render("LFM Lattice", True, WHITE)
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += 40
        
        # Stats
        stats = [
            f"Steps: {self.step_count}",
            f"Energy: {self.energy_history[-1]:.3e}" if self.energy_history else "Energy: 0.0",
            "",
            f"Chi: {self.chi:.2f}",
            f"Damping: {self.gamma_damp:.3f}",
            f"dt: {self.dt:.3f}",
            "",
            "Controls:",
            "SPACE: Pause",
            "R: Reset",
            "Click: Add pulse",
            "Shift+Click: Line",
            "",
            "Q/A: Chi ±0.1",
            "W/S: Damp ±0.01",
            "E/D: dt ±0.01",
            "+/-: Amplitude",
            "",
            "3: Toggle 3D view",
            "F1: Double Slit",
        ]
        
        # Add scenario-specific controls
        if self.scenario_mode:
            stats.append("")
            stats.append(f"SCENARIO MODE")
            stats.append(f"{self.scenario_name}")
            stats.append(f"Frame: {self.scenario_frame_index}/{len(self.scenario_frames)}")
            stats.append("")
            stats.append("←/→: Step frame")
            stats.append("[/]: Speed ±")
            stats.append(f"Speed: {self.playback_speed}x")
            stats.append("ESC: Exit scenario")
        
        for line in stats:
            text = self.small_font.render(line, True, WHITE)
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 20
            
        # Status
        if self.scenario_mode:
            if self.scenario_playing:
                status = self.font.render("PLAYING", True, GREEN)
            else:
                status = self.font.render("PAUSED", True, RED)
            self.screen.blit(status, (panel_x, y_offset + 20))
        elif self.paused:
            status = self.font.render("PAUSED", True, RED)
            self.screen.blit(status, (panel_x, y_offset + 20))
            
    def handle_events(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Disable clicks in scenario mode
                if not self.scenario_mode and event.pos[0] < WINDOW_SIZE:
                    # Convert screen coords to grid coords
                    grid_x = int((event.pos[0] / WINDOW_SIZE) * GRID_SIZE)
                    grid_y = int((event.pos[1] / WINDOW_SIZE) * GRID_SIZE)
                    
                    # Check if shift is held (line pulse)
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self.add_line_pulse(grid_x)
                    else:
                        self.add_gaussian_pulse(grid_x, grid_y)
                        
            elif event.type == pygame.KEYDOWN:
                # Scenario mode controls
                if self.scenario_mode:
                    if event.key == pygame.K_SPACE:
                        self.scenario_playing = not self.scenario_playing
                    elif event.key == pygame.K_LEFT:
                        # Step backward
                        self.load_scenario_frame(self.scenario_frame_index - 1)
                        self.scenario_playing = False
                    elif event.key == pygame.K_RIGHT:
                        # Step forward
                        self.load_scenario_frame(self.scenario_frame_index + 1)
                        self.scenario_playing = False
                    elif event.key == pygame.K_LEFTBRACKET:
                        # Decrease speed
                        self.playback_speed = max(1, self.playback_speed - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        # Increase speed
                        self.playback_speed = min(10, self.playback_speed + 1)
                    elif event.key == pygame.K_ESCAPE:
                        self.exit_scenario_mode()
                    elif event.key == pygame.K_3:
                        self.toggle_3d_view()
                        
                # Normal mode controls
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_q:
                    self.chi += 0.1
                elif event.key == pygame.K_a:
                    self.chi = max(0.0, self.chi - 0.1)
                elif event.key == pygame.K_w:
                    self.gamma_damp += 0.01
                elif event.key == pygame.K_s:
                    self.gamma_damp = max(0.0, self.gamma_damp - 0.01)
                elif event.key == pygame.K_e:
                    self.dt += 0.01
                elif event.key == pygame.K_d:
                    self.dt = max(0.01, self.dt - 0.01)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.pulse_amplitude *= 1.2
                elif event.key == pygame.K_MINUS:
                    self.pulse_amplitude /= 1.2
                elif event.key == pygame.K_3:
                    self.toggle_3d_view()
                        
                # Global controls (work in both modes)
                if event.key == pygame.K_F1:
                    self.load_double_slit_scenario()
                    
        return True
        
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            
            # Step physics or advance scenario
            if self.scenario_mode:
                # Scenario playback mode
                if self.scenario_playing:
                    # Advance by playback_speed frames
                    new_index = self.scenario_frame_index + self.playback_speed
                    if new_index >= len(self.scenario_frames):
                        # Loop back to start
                        new_index = 0
                    self.load_scenario_frame(new_index)
            else:
                # Normal interactive mode
                if not self.paused:
                    for _ in range(STEPS_PER_FRAME):
                        self.step_physics()
                    
            # Render 2D pygame view
            self.screen.fill(BLACK)
            self.render_field()
            self.render_ui()
            self.render_energy_graph()
            pygame.display.flip()
            
            # Update 3D view if enabled (less frequently for performance)
            if self.show_3d and self.step_count % 5 == 0:  # Update every 5 frames
                self.update_3d_view()
            
            self.clock.tick(FPS_TARGET)
            
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    print("="*60)
    print("LFM INTERACTIVE LATTICE PLAYGROUND")
    print("="*60)
    print("\nInteractive Mode:")
    print("  Click to add energy pulses")
    print("  Use keyboard to adjust parameters")
    print("  See on-screen controls for details")
    print("\nScenarios:")
    print("  Press F1 to load 'The Double Slit Experiment'")
    print("="*60 + "\n")
    
    lattice = InteractiveLattice()
    lattice.run()
