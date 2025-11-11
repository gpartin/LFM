# Deploy to emergentphysicslab.com - Step by Step

**Platform**: Vercel  
**Time Required**: 15-20 minutes  
**Date**: November 10, 2025

---

## Prerequisites Checklist

Before starting, ensure:
- ✅ You have access to GitHub account (gpartin/LFM)
- ✅ You own domain: emergentphysicslab.com
- ✅ You can access domain registrar DNS settings
- ✅ Local build works: `npm run build` succeeds

---

## Step 1: Verify Local Build

**Why**: Catch errors before deploying.

```powershell
cd c:\LFM\workspace\website
npm run build
```

**Expected Output:**
```
✓ Generating static pages (17/17)
✓ Compiled successfully
Route (app)                              Size     First Load JS
┌ ○ /                                    5.2 kB    120 kB
├ ○ /about                               1.8 kB    118 kB
└ ○ /experiments                         2.1 kB    119 kB
```

**If you see errors**: Stop and fix them before proceeding.

**✅ Success Criteria**: No red error messages, build completes.

---

## Step 2: Commit Documentation Changes

**Why**: Clean up Netlify references, organize docs.

```powershell
cd c:\LFM
git status  # Review changes
git add workspace/website/
git commit -m "Website deployment preparation: Remove Netlify, organize docs, add step control"
git push origin main
```

**✅ Success Criteria**: `git push` completes without errors.

**Verify**: Visit https://github.com/gpartin/LFM/tree/main/workspace/website

---

## Step 3: Create Vercel Account

**Why**: One-time setup to connect GitHub.

1. **Go to**: https://vercel.com/signup
2. **Click**: "Continue with GitHub"
3. **Authorize**: Vercel to access your GitHub
4. **Result**: Redirected to Vercel dashboard

**✅ Success Criteria**: You see Vercel dashboard with "Add New Project" button.

---

## Step 4: Import LFM Repository

**Why**: Tell Vercel which GitHub repo to deploy.

1. **Click**: "Add New..." → "Project"
2. **Find**: `LFM` in repository list
3. **Click**: "Import" button next to LFM

**If you don't see LFM:**
- Click "Adjust GitHub App Permissions"
- Grant Vercel access to the repository
- Return to import screen

**✅ Success Criteria**: You see the "Configure Project" screen.

---

## Step 5: Configure Project Settings

**Why**: Tell Vercel where your Next.js app lives.

### Root Directory ⚠️ IMPORTANT
- Click "Edit" next to "Root Directory"
- Enter: `workspace/website`
- Click checkmark to save

### Other Settings (Auto-Detected - Leave As-Is)
- **Framework Preset**: Next.js ✅
- **Build Command**: `npm run build` ✅
- **Output Directory**: `.next` ✅
- **Install Command**: `npm install` ✅

### Environment Variables (Skip for Now)
- We'll add these after first deployment

**Click**: Big blue "Deploy" button at bottom

**✅ Success Criteria**: Build log starts streaming.

---

## Step 6: Watch First Build (2-3 minutes)

**Why**: Ensure build completes successfully.

**What You'll See:**
```
Cloning repository...
Installing dependencies...
Running "npm install"...
Running "npm run build"...
> prebuild
> Generating experiments from test configs...
✓ Generated 104 experiments
Building application...
✓ Generating static pages (17/17)
Uploading build outputs...
✓ Deployment ready!
```

**When Complete:**
- Big confetti animation 🎉
- URL like: `lfm-abc123.vercel.app`
- "Visit" button appears

**Click "Visit"** to see your live site!

**✅ Success Criteria**: Website loads, you see home page with experiments.

---

## Step 7: Test Your Deployment

**Why**: Verify everything works before adding custom domain.

**Manual Tests:**
1. **Home Page**: Should load instantly
2. **Click**: "Showcase Experiments" → Pick one → Should show 3D simulation
3. **Click**: "All Research Experiments" → Should show 104 tests
4. **Pick Any Test**: (e.g., REL-01) → Click "Step Forward" button → Should advance simulation
5. **GRAV-09**: Should show yellow warning banner

**✅ Success Criteria**: All pages load, simulations run, controls work.

**Common Issues:**
- **Blank page**: Check browser console (F12) for errors
- **No 3D graphics**: WebGPU not supported (expected on some browsers)
- **Slow loading**: Normal for first load, cache warms up

---

## Step 8: Add Custom Domain

**Why**: Make site accessible at emergentphysicslab.com.

1. **In Vercel**: Click your project name (top left)
2. **Click**: "Settings" tab
3. **Click**: "Domains" (left sidebar)
4. **In the text box**, enter: `emergentphysicslab.com`
5. **Click**: "Add"

Vercel will show: "Configure DNS Records"

**✅ Success Criteria**: You see DNS instructions with specific values.

---

## Step 9: Configure DNS Records

**Why**: Point your domain to Vercel's servers.

### Find Your Domain Registrar
Where did you buy emergentphysicslab.com? (GoDaddy, Namecheap, Google Domains, etc.)

### Add DNS Records

**Log into your domain registrar**, find DNS settings, and add:

#### Record 1: Root Domain (A Record)
```
Type: A
Name: @ (or leave blank)
Value: 76.76.21.21
TTL: 3600 (or Auto)
```

#### Record 2: WWW Subdomain (CNAME Record)
```
Type: CNAME
Name: www
Value: cname.vercel-dns.com
TTL: 3600 (or Auto)
```

**Save Changes**

**✅ Success Criteria**: DNS records saved in registrar dashboard.

---

## Step 10: Wait for DNS Propagation

**Why**: DNS changes take time to spread worldwide.

**Expected Wait Time**: 10-60 minutes (sometimes up to 24 hours)

### Check Propagation Status

**Online Tool**: https://www.whatsmydns.net/#A/emergentphysicslab.com

**PowerShell Command**:
```powershell
nslookup emergentphysicslab.com
```

**When Ready:**
- Vercel dashboard will show green checkmark next to domain
- `emergentphysicslab.com` will resolve to `76.76.21.21`

**✅ Success Criteria**: 
- Visit https://emergentphysicslab.com
- Site loads (same as vercel.app URL)

---

## Step 11: Add Environment Variables (Optional)

**Why**: Enable production features and tracking.

1. **In Vercel**: Settings → Environment Variables
2. **Add each variable** (click "Add Another" between each):

```
NEXT_PUBLIC_SITE_URL=https://emergentphysicslab.com
NEXT_PUBLIC_CONTACT_EMAIL=research@emergentphysicslab.com
NEXT_PUBLIC_ENABLE_WEBGPU=true
NEXT_PUBLIC_ENABLE_ANALYTICS=false
NEXT_PUBLIC_ENABLE_VALIDATION_SIGNING=false
```

3. **Select**: Production, Preview, Development (all three)
4. **Click**: "Save"

### Trigger Redeploy
Changes require rebuild:
1. **Click**: "Deployments" tab
2. **Find**: Latest deployment
3. **Click**: "..." menu → "Redeploy"
4. **Confirm**: "Redeploy"

**✅ Success Criteria**: New deployment completes successfully.

---

## Step 12: Submit to Google

**Why**: Get indexed in search results faster.

### Add Sitemap to Google Search Console

1. **Go to**: https://search.google.com/search-console
2. **Add Property**: emergentphysicslab.com
3. **Verify Ownership**: Choose DNS verification
   - Add TXT record to your domain DNS
   - Wait for verification (can take a few hours)
4. **Submit Sitemap**: 
   - Click "Sitemaps" (left sidebar)
   - Enter: `https://emergentphysicslab.com/sitemap.xml`
   - Click "Submit"

**Expected Indexing Time**: 3-7 days for first pages to appear in Google.

**✅ Success Criteria**: Sitemap submitted, no errors in Search Console.

---

## 🎉 Deployment Complete!

Your website is now live at **https://emergentphysicslab.com**

### What Happens Next (Automatic)

**Every time you push to GitHub:**
1. Vercel detects the push
2. Automatically builds the new version
3. Deploys to production (30-60 seconds)
4. You get an email notification

**No manual deployment needed ever again!**

---

## Post-Deployment Checklist

Within 24 hours of going live:

- [ ] Test all 104 research experiments load
- [ ] Verify mobile responsive design
- [ ] Check page load speed (<3 seconds)
- [ ] Test step controls work on at least 5 experiments
- [ ] Verify GRAV-09 skip warning displays
- [ ] Check sitemap.xml accessible
- [ ] Check robots.txt accessible
- [ ] Verify social media cards (share on Twitter/LinkedIn to test)
- [ ] Monitor Vercel Analytics for traffic
- [ ] Set up uptime monitoring (https://uptimerobot.com - free)

---

## Troubleshooting Common Issues

### Build Fails: "Cannot find module"

**Issue**: Missing dependency or path error.

**Fix**:
```powershell
cd c:\LFM\workspace\website
rm -rf node_modules package-lock.json
npm install
npm run build  # Test locally first
```

If local build works but Vercel fails:
- Check Root Directory is set to `workspace/website`
- Verify `package.json` is committed to git

---

### DNS Not Working After 24 Hours

**Check DNS records are correct**:
```powershell
nslookup emergentphysicslab.com
```

Should return: `76.76.21.21`

**If wrong IP**:
- Log back into domain registrar
- Verify A record points to `76.76.21.21`
- Delete any conflicting records
- Save and wait another hour

---

### WebGPU Not Working

**Expected Behavior**:
- Works: Chrome 113+, Edge 113+
- Doesn't work: Firefox, Safari (will show CPU fallback message)

**If not working in Chrome**:
- Check browser console (F12) for CORS errors
- Verify `vercel.json` has correct headers (already set)
- Try incognito mode (disables extensions)

---

### Slow Page Load

**First load is always slower** (cold start).

**Check performance**:
1. **In Vercel**: Analytics tab → Speed Insights
2. **Expected**: <3s Time to First Byte
3. **If slower**: Check bundle size in build log

**Optimization (future)**:
- Enable Vercel Image Optimization
- Add Redis caching for API routes
- Implement service worker for offline support

---

## Future Enhancements (Phase 2)

Once site is stable, add:

### 1. Validation Signatures
- Add Vercel Postgres database
- Store user experiment validations
- Public leaderboard of validators

### 2. User Accounts
- NextAuth.js authentication
- Save experiment history per user
- Email notifications for new experiments

### 3. Analytics
- Enable Vercel Analytics (free tier)
- Track most popular experiments
- Monitor WebGPU vs CPU fallback ratio

---

## Support

**Vercel Issues**: https://vercel.com/support  
**DNS Issues**: Contact your domain registrar  
**Website Bugs**: Open issue on GitHub

---

**Last Updated**: November 10, 2025
