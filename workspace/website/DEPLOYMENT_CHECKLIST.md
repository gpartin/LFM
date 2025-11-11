# Pre-Deployment Checklist

**Target**: emergentphysicslab.com  
**Date**: November 10, 2025

## ✅ Code Ready

- [x] Next.js 14 app configured
- [x] All 105 research experiments generated
- [x] Step control implemented for all canvas types
- [x] GRAV-09 skip warning banner
- [x] TypeScript compiles without errors
- [x] Tests passing (37+ tests)

## ✅ Deployment Files Created

- [x] `vercel.json` - Deployment configuration
- [x] `.env.example` - Environment template
- [x] `public/robots.txt` - Search engine directives
- [x] `src/app/sitemap.ts` - Dynamic sitemap (105+ pages)
- [x] Enhanced SEO metadata in `layout.tsx`
- [x] JSON-LD structured data for rich results

## 🔄 To Do Before Deploy

### Critical (Must Do):
- [ ] **Test local build**: `npm run build` succeeds
- [ ] **Test local production**: `npm start` works
- [ ] **Verify sitemap**: `http://localhost:3000/sitemap.xml` renders
- [ ] **Check robots.txt**: `http://localhost:3000/robots.txt` exists

### Important (Should Do):
- [ ] Create OG image (`public/og-image.png`) - 1200×630px
- [ ] Create favicon set (`public/favicon.ico`, etc.)
- [ ] Create logo (`public/logo.png`)
- [ ] Review all text for typos
- [ ] Test on mobile device

### Optional (Nice to Have):
- [ ] Add Google Analytics ID (can add later)
- [ ] Set up Plausible Analytics (can add later)
- [ ] Create Twitter account @EmergentPhysics
- [ ] Prepare initial social media posts

## 🚀 Deployment Steps

### 1. Local Build Test
```bash
cd c:\LFM\workspace\website
npm run build
npm start
# Visit http://localhost:3000 and test
```

### 2. Push to GitHub
```bash
cd c:\LFM
git add workspace/website/
git commit -m "Deploy: Add Vercel config, SEO optimization, and deployment docs"
git push origin main
```

### 3. Connect Vercel
1. Visit https://vercel.com/new
2. Import GitHub repo: `gpartin/LFM`
3. Root directory: `workspace/website`
4. Environment variables: (copy from `.env.example`)
   ```
   NEXT_PUBLIC_SITE_URL=https://emergentphysicslab.com
   NEXT_PUBLIC_CONTACT_EMAIL=research@emergentphysicslab.com
   NEXT_PUBLIC_ENABLE_WEBGPU=true
   ```
5. Click **Deploy**

### 4. Configure Domain
**In Vercel Dashboard:**
1. Settings → Domains
2. Add: `emergentphysicslab.com`
3. Add: `www.emergentphysicslab.com`

**In Domain Registrar (GoDaddy/Namecheap/etc):**
```
Type    Name    Value               TTL
A       @       76.76.21.21         3600
CNAME   www     cname.vercel-dns.com 3600
```

### 5. Wait for DNS (10-60 minutes)
Check: https://www.whatsmydns.net/#A/emergentphysicslab.com

### 6. Verify Deployment
- [ ] https://emergentphysicslab.com loads
- [ ] https://www.emergentphysicslab.com redirects correctly
- [ ] https://emergentphysicslab.com/sitemap.xml works
- [ ] https://emergentphysicslab.com/robots.txt works
- [ ] https://emergentphysicslab.com/experiments loads
- [ ] Test a research experiment: https://emergentphysicslab.com/experiments/REL-01
- [ ] Verify GRAV-09 warning: https://emergentphysicslab.com/experiments/GRAV-09

## 📊 Post-Deployment (Day 1)

### Submit to Search Engines
- [ ] **Google Search Console**
  1. https://search.google.com/search-console
  2. Add property: `emergentphysicslab.com`
  3. Verify via DNS or HTML tag
  4. Submit sitemap: `https://emergentphysicslab.com/sitemap.xml`

- [ ] **Bing Webmaster Tools**
  1. https://www.bing.com/webmasters
  2. Add site
  3. Submit sitemap

### Archive Site
- [ ] Internet Archive: https://web.archive.org/save/emergentphysicslab.com
- [ ] Archive.today: http://archive.today/ (submit URL)

### Test Performance
- [ ] Lighthouse audit: https://pagespeed.web.dev/
  - Target: >90 Performance
  - Target: 100 Accessibility
  - Target: 100 Best Practices
  - Target: 100 SEO

### Monitor
- [ ] Vercel Dashboard → Analytics (check for errors)
- [ ] Check Vercel logs for any build warnings
- [ ] Test from different devices/browsers

## 📈 Post-Deployment (Week 1)

- [ ] Monitor Google Search Console for crawl errors
- [ ] Check which pages get indexed first
- [ ] Share on physics forums (with permission)
- [ ] Add to academic profiles (ResearchGate, etc.)
- [ ] Monitor uptime (set up UptimeRobot - free)

## 🔧 If Build Fails

### Common Issues:

**Issue**: Cannot find `../tools/generate_website_experiments.js`
```bash
# Workaround: Commit generated file
cd c:\LFM\workspace\website
git add src/data/research-experiments-generated.ts -f
git commit -m "Add generated experiments for deployment"
git push
```

**Issue**: TypeScript errors
```bash
# Run type check locally first
npm run type-check
# Fix errors before deploying
```

**Issue**: Build timeout
```bash
# Increase Vercel timeout (Pro plan) or optimize build
# Split large files, lazy load components
```

## 📞 Support Contacts

- **Vercel Support**: https://vercel.com/help
- **Domain Issues**: Your registrar support
- **Technical**: latticefieldmediumresearch@gmail.com

## 🎯 Success Criteria

**Immediate (Day 1):**
- ✅ Site accessible at emergentphysicslab.com
- ✅ All pages load without errors
- ✅ Mobile responsive
- ✅ HTTPS enabled

**Week 1:**
- ✅ Sitemap submitted to search engines
- ✅ At least homepage indexed by Google
- ✅ No critical errors in Search Console
- ✅ <3s page load time

**Month 1:**
- ✅ 50+ pages indexed
- ✅ Appears in search for "Klein Gordon lattice simulation"
- ✅ 100+ unique visitors
- ✅ Featured in at least 1 physics forum/discussion

---

**Ready to deploy?** Start with Step 1 (Local Build Test) above! 🚀
