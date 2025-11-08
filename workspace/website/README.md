# Emergent Physics Lab - Interactive Website

**Live Site**: [emergentphysicslab.com](https://emergentphysicslab.com)

This website demonstrates how fundamental forces emerge from the **Lattice Field Medium (LFM)** framework through interactive 3D experiments.

## What This Is

An educational platform that runs **real physics simulations** in your browser:
- **WebGPU backend**: Runs actual Klein-Gordon lattice simulation (64³ grid)
- **3D visualization**: Three.js rendering of particles, fields, and lattice structure
- **Real-time interaction**: Adjust parameters and watch physics emerge

## The Physics

All experiments are based on the modified Klein-Gordon equation:

```
∂²E/∂t² = c²∇²E − χ²(x,t)E
```

From this single equation, we observe:
- **Gravity** emerges from χ field gradients
- **Relativity** emerges from lattice structure (Lorentz invariance)
- **Energy conservation** maintained to <0.01% drift
- **Orbital mechanics** from first principles (no Newton's laws needed)

## Project Structure

```
website/
├── src/
│   ├── app/              # Next.js app router pages
│   ├── components/       # React components
│   │   ├── experiments/  # Individual experiment UIs
│   │   ├── canvas/       # Three.js 3D components
│   │   └── ui/           # Shared UI components
│   ├── physics/          # Physics engine (THE CORE)
│   │   ├── core/         # LFM lattice simulation
│   │   │   ├── lattice-webgpu.ts    # GPU compute shaders
│   │   │   ├── lattice-cpu.ts       # CPU fallback
│   │   │   └── klein-gordon.ts      # Equation implementation
│   │   ├── forces/       # Emergent phenomena
│   │   │   ├── gravity.ts
│   │   │   └── orbits.ts
│   │   └── utils/        # Math helpers
│   └── lib/              # Utilities
├── public/               # Static assets
└── docs/                 # Technical documentation
```

## Key Features

### 1. Adaptive Backend Selection
The site automatically detects your hardware and selects the best physics backend:

- **WebGPU** (Chrome 113+, GPU required): Full LFM lattice simulation
- **WebGL2** (Any modern browser): Approximate simulation
- **CPU Fallback** (All browsers): Simplified Newtonian physics

**Important**: If running CPU fallback, a prominent warning displays:
> ⚠️ **Running simplified physics (CPU mode)** - Not authentic LFM simulation. Upgrade browser for real experience.

### 2. Binary Orbit Experiment
Watch two masses orbit each other with gravity emerging from the chi field:

- Real-time parameter adjustment (mass ratio, distance, chi strength)
- Energy conservation tracking (live metrics)
- View modes: Particles only, Chi field visualization, Full lattice
- Orbital trail rendering

## Research Links

- **OSF Project**: [10.17605/OSF.IO/6AGN8](https://osf.io/6agn8)
- **Zenodo Archive**: [10.5281/zenodo.17536484](https://zenodo.org/records/17536484)
- **GitHub Repository**: [github.com/gpartin/LFM](https://github.com/gpartin/LFM)

## Development

### Prerequisites
- Node.js 18+
- Modern browser with WebGPU support (recommended: Chrome 113+)

### Setup
```bash
cd workspace/website
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production
```bash
npm run build
npm start
```

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **3D Rendering**: Three.js + React Three Fiber
- **Physics**: Custom WebGPU compute shaders
- **Styling**: Tailwind CSS
- **Deployment**: Netlify

## How the Physics Works

### WebGPU Backend (Authentic LFM)

1. **Lattice Initialization**: Create 64×64×64 grid of energy values
2. **Klein-Gordon Evolution**: Each frame, GPU computes:
   ```glsl
   // WGSL compute shader
   E_next = 2*E_curr - E_prev + dt*dt * (c*c*laplacian - chi*chi*E_curr)
   ```
3. **Force Extraction**: Calculate chi field gradients → emergent gravity
4. **Particle Tracking**: Update particle positions based on field forces
5. **Visualization**: Render lattice state to Three.js scene

### CPU Fallback (Simplified)

For users without WebGPU, we fall back to Newtonian gravity:
```typescript
// Clearly marked as approximation, not real LFM
const force = G * m1 * m2 / (r * r);
```

## Why This Matters

This is not just a visualization - it's **proof of concept**:
- Shows that fundamental physics CAN emerge from simple rules
- Demonstrates energy conservation in real-time
- Makes complex physics accessible and interactive
- All code is open source and inspectable

## License

Code: MIT License (website functionality)
Physics/Content: CC BY-NC-ND 4.0 (research content)

## Author

Greg Partin - [Research Profile](https://osf.io/6agn8)

---

**Note**: This website runs real physics simulations. Performance depends on your hardware. For best experience, use a device with WebGPU support.
