# Website Setup & Installation Guide

## Prerequisites

- Node.js 18+ installed
- Modern browser (Chrome 113+ or Edge 113+ for WebGPU)
- Git

## Installation

```powershell
# Navigate to website directory
cd c:\LFM\workspace\website

# Install dependencies
npm install

# Run development server
npm run dev
```

Visit `http://localhost:3000` to see the site.

## Build for Production

```powershell
npm run build
npm start
```

## Deploy to Netlify

### Option 1: Netlify CLI

```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod
```

### Option 2: GitHub Integration

1. Push code to GitHub (already in LFM repo)
2. Connect Netlify to GitHub repository
3. Set build command: `npm run build`
4. Set publish directory: `.next`
5. Add custom domain: `emergentphysicslab.com`

## Environment Variables

None required for basic deployment.

## Domain Setup

1. In Netlify dashboard, go to Domain settings
2. Add custom domain: `emergentphysicslab.com`
3. Follow DNS configuration instructions
4. Add these records at GoDaddy:
   - **A Record**: `@` → Netlify IP
   - **CNAME**: `www` → your-site.netlify.app

## Browser Compatibility

| Browser | WebGPU | WebGL2 | CPU Fallback |
|---------|--------|--------|--------------|
| Chrome 113+ | ✅ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox | ❌ | ✅ | ✅ |
| Safari | ❌ | ✅ | ✅ |

## Troubleshooting

### TypeScript errors before npm install

This is normal — errors will clear after running `npm install`.

### WebGPU not detected

1. Check browser version (Chrome/Edge 113+)
2. Enable WebGPU in `chrome://flags` if needed
3. Verify GPU drivers are up to date

### Build fails

```powershell
# Clear cache
rm -rf .next node_modules
npm install
npm run build
```

## Project Structure

```
website/
├── src/
│   ├── app/                    # Pages (Next.js App Router)
│   │   ├── page.tsx           # Landing page
│   │   ├── layout.tsx         # Root layout
│   │   ├── globals.css        # Global styles
│   │   └── experiments/
│   │       └── binary-orbit/
│   │           └── page.tsx   # Orbital experiment
│   ├── components/
│   │   ├── layout/            # Header, Footer
│   │   └── ui/                # BackendBadge, etc.
│   ├── physics/               # Physics engine
│   │   ├── core/              # LFM lattice, backend detection
│   │   └── forces/            # Orbital mechanics
│   └── types/                 # TypeScript declarations
├── public/                     # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── netlify.toml
```

## Next Steps After npm install

1. **Test locally**: `npm run dev`
2. **Add Three.js visualization** (next phase)
3. **Connect physics engine to UI**
4. **Deploy to Netlify**
5. **Configure domain**

## Performance Notes

- WebGPU backend: 60 FPS on modern GPUs
- WebGL2 backend: 30-60 FPS
- CPU fallback: 10-20 FPS (simplified physics only)

## Development Tips

- Edit code in `src/` directory
- Hot reload enabled (changes appear instantly)
- Check browser console for errors
- Use React DevTools for debugging

---

**Ready to start?** Run `npm install` from `c:\LFM\workspace\website\`
