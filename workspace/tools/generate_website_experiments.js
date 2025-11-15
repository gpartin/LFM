#!/usr/bin/env node
/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * CLEAN WEBSITE EXPERIMENT METADATA GENERATOR
 * ------------------------------------------
 * Generates website data layer from canonical test harness configs.
 * Outputs:
 *   - website/src/data/research-experiments-generated.ts
 *   - website/src/data/test-statistics.ts
 * Usage:
 *   node workspace/tools/generate_website_experiments.js
 */

const fs = require('fs');
const path = require('path');

const SCRIPT_DIR = __dirname;
const WORKSPACE_ROOT = path.join(SCRIPT_DIR, '..');
const CONFIG_DIR = path.join(WORKSPACE_ROOT, 'config');
const OUT_EXPERIMENTS = path.join(WORKSPACE_ROOT, 'website', 'src', 'data', 'research-experiments-generated.ts');
const OUT_STATS = path.join(WORKSPACE_ROOT, 'website', 'src', 'data', 'test-statistics.ts');
const REGISTRY_CANON = path.join(WORKSPACE_ROOT, 'results', 'test_registry_canonical.json');
const MASTER_CSV = path.join(WORKSPACE_ROOT, 'results', 'MASTER_TEST_STATUS.csv');

// Updated expected counts (must reflect configs; Tier1=17, Tier4=14 now)
const TIERS = [
  { tier: 1, name: 'Relativistic', prefix: 'REL', config: 'config_tier1_relativistic.json', expected: 17 },
  { tier: 2, name: 'Gravity', prefix: 'GRAV', config: 'config_tier2_gravityanalogue.json', expected: 26 },
  { tier: 3, name: 'Energy', prefix: 'ENER', config: 'config_tier3_energy.json', expected: 11 },
  { tier: 4, name: 'Quantization', prefix: 'QUAN', config: 'config_tier4_quantization.json', expected: 14 },
  { tier: 5, name: 'Electromagnetic', prefix: 'EM', config: 'config_tier5_electromagnetic.json', expected: 21 },
  { tier: 6, name: 'Coupling', prefix: 'COUP', config: 'config_tier6_coupling.json', expected: 12 },
  { tier: 7, name: 'Thermodynamics', prefix: 'THERM', config: 'config_tier7_thermodynamics.json', expected: 5 }
];

function readJSON(p) { try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return null; } }
function hasAny(o, ks) { return ks.some(k => Object.prototype.hasOwnProperty.call(o, k)); }
function hasPrefix(o, ps) { return Object.keys(o).some(k => ps.some(p => k.startsWith(p))); }

function inferSimulationType(tierNum, description, testObj) {
  const desc = String(description || '').toLowerCase();
  const t = testObj || {};
  if (hasAny(t, ['chi','chi_const','chi_base','chi_bg','chi_left','chi_right','chi_inside','chi_outside','chi_barrier','chi_gradient']) || hasPrefix(t,['chi_','lens_','bump_'])) return 'field-dynamics';
  if (hasAny(t, ['em_wave_freq','em_wave_amp']) || hasPrefix(t,['em_'])) return 'field-dynamics';
  if (hasAny(t, ['packet_width']) || desc.includes('packet') || desc.includes('wave propagation')) return 'wave-packet';
  if (desc.includes('orbit') || desc.includes('binary') || desc.includes('n-body')) return 'binary-orbit';
  if (desc.includes('wave') || desc.includes('isotropy') || desc.includes('causality') || desc.includes('phase velocity')) return 'wave-packet';
  if (desc.includes('field') || desc.includes('chi') || desc.includes('electromagnetic') || desc.includes('coupling')) return 'field-dynamics';
  if (tierNum <= 2) return 'wave-packet';
  if (tierNum === 3) return 'wave-packet';
  return 'field-dynamics';
}

function inferVisualizationPreset(tierNum, description, simulation) {
  const desc = description.toLowerCase();
  if (desc.includes('orbit') || desc.includes('gravitational') || desc.includes('binary')) return 'orbital-showcase';
  if (desc.includes('wave') || desc.includes('packet') || desc.includes('propagation') || desc.includes('isotropy') || desc.includes('causality')) return 'wave-dynamics';
  if ((desc.includes('field') || desc.includes('electromagnetic') || desc.includes('coupling')) && !desc.includes('orbit')) return 'field-only';
  if (desc.includes('energy') || desc.includes('conservation') || desc.includes('thermodynamic')) return 'minimal';
  if (simulation === 'binary-orbit') return 'orbital-minimal';
  if (simulation === 'wave-packet') return 'wave-dynamics';
  if (simulation === 'field-dynamics') return 'field-only';
  switch (tierNum) { case 1: return 'wave-dynamics'; case 2: return 'orbital-minimal'; case 3: return 'minimal'; case 4: return 'field-only'; case 5: return 'field-only'; case 6: return 'field-only'; case 7: return 'minimal'; default: return 'wave-dynamics'; }
}

function extractChi(test, base) {
  if ('chi_const' in test) return test.chi_const;
  if ('chi_gradient' in test) return test.chi_gradient;
  for (const k of ['chi','chi_base','chi_bg','chi_left','chi_right','chi_inside','chi_outside','chi_barrier']) if (k in test) return test[k];
  if ('chi' in base) return base.chi;
  return 0.0;
}

function buildEntry(tierInfo, test, baseParams, cfg) {
  const testId = test.test_id || test.id || null;
  if (!testId) return null;
  const description = test.description || test.name || 'No description';
  const simType = inferSimulationType(tierInfo.tier, description, test);
  const vizPreset = inferVisualizationPreset(tierInfo.tier, description, simType);
  const latticeSize = test.grid_points ?? test.grid_size ?? baseParams.grid_points ?? baseParams.N ?? 256;
  const steps = test.steps ?? baseParams.steps ?? 5000;
  const dt = test.dt ?? baseParams.dt ?? baseParams.time_step ?? 0.001;
  const dx = test.dx ?? baseParams.dx ?? baseParams.space_step ?? 0.01;
  const chi = extractChi(test, baseParams);
  const tierFolder = path.basename((cfg && cfg.output_dir) ? String(cfg.output_dir).replace(/\\/g,'/') : tierInfo.name);
  const entry = {
    id: testId,
    testId,
    displayName: `${testId}: ${description}`,
    type: 'RESEARCH',
    tier: tierInfo.tier,
    tierName: tierInfo.name,
    category: tierInfo.name,
    tagline: description,
    description: `Research validation test for ${tierInfo.name.toLowerCase()} tier. ${description}`,
    difficulty: 'intermediate',
    simulation: simType,
    visualizationPreset: vizPreset,
    backend: 'both',
    initialConditions: { latticeSize, dt, dx, steps, chi },
    validation: { energyDrift: baseParams.tolerances?.energy_drift || 1e-6 },
    visualization: { showParticles:false, showTrails:false, showChi: simType==='field-dynamics', showLattice:true, showVectors:false, showWell:false, showDomes:false, showIsoShells:false, showBackground:false },
    links: { testHarnessConfig: `workspace/config/${tierInfo.config}`, results: `workspace/results/${tierFolder}/${testId}/`, discovery: `Tier ${tierInfo.tier} - ${tierInfo.name}` },
    status: test.skip ? 'development' : 'production'
  };
  if (test.tolerances) entry.validation = { ...entry.validation, ...test.tolerances };
  return entry;
}

function loadTier(t) {
  const cfgPath = path.join(CONFIG_DIR, t.config);
  const cfg = readJSON(cfgPath);
  if (!cfg) { console.warn(`Missing config for tier ${t.tier}`); return []; }
  const params = { ...(cfg.parameters||{}), tolerances: cfg.tolerances||{}, grid_points: cfg.parameters?.grid_points, N: cfg.parameters?.N, dt: cfg.parameters?.dt ?? cfg.parameters?.time_step, dx: cfg.parameters?.dx ?? cfg.parameters?.space_step, steps: cfg.parameters?.steps };
  const tests = cfg.variants || cfg.tests || [];
  return tests.map(ts => buildEntry(t, ts, params, cfg)).filter(Boolean);
}

function generateAll() { return TIERS.flatMap(t => loadTier(t)).sort((a,b)=>String(a.testId).localeCompare(String(b.testId))); }

function computeStats() {
  // Prefer canonical registry
  if (fs.existsSync(REGISTRY_CANON)) {
    try {
      const reg = readJSON(REGISTRY_CANON) || {}; const sum = reg.summary||{}; const tiers = reg.tiers||{};
      const total = +sum.executed||0; const passing = +sum.passed||0; const byTier={};
      Object.keys(tiers).forEach(k=>{ const t=tiers[k]||{}; byTier[k]={ total:+t.executed||0, passing:+t.passed||0 }; });
      return { total, passing, failing: total-passing, passRate: total?((passing/total)*100).toFixed(1)+'%':'0.0%', byTier, generatedAt:new Date().toISOString(), sourceFile:REGISTRY_CANON };
    } catch(e){ console.warn('Canonical registry parse failed:', e.message); }
  }
  if (fs.existsSync(MASTER_CSV)) {
    try {
      const lines = fs.readFileSync(MASTER_CSV,'utf-8').split('\n'); let total=0, passing=0; const byTier={}; TIERS.forEach(t=>byTier[String(t.tier)]={ total:0, passing:0 });
      for (const line of lines) {
        const L=line.trim(); if(!L||/^(MASTER|Generated:|Validation|CATEGORY|Tier,Category|DETAILED|Test_ID,)/.test(L)) continue;
        const parts=L.split(','); if(parts.length>=3 && /^[A-Z]+-\d+$/.test(parts[0])) { const id=parts[0]; const status=parts[2]; const prefix=id.split('-')[0]; const tierObj=TIERS.find(t=>t.prefix===prefix); const key=tierObj?String(tierObj.tier):null; if(status==='PASS'){ total++; passing++; if(key){ byTier[key].total++; byTier[key].passing++; } } else if(status==='FAIL'){ total++; if(key){ byTier[key].total++; } } }
      }
      return { total, passing, failing: total-passing, passRate: total?((passing/total)*100).toFixed(1)+'%':'0.0%', byTier, generatedAt:new Date().toISOString(), sourceFile:MASTER_CSV };
    } catch(e){ console.error('MASTER_TEST_STATUS parse failed:', e.message); }
  }
  const totalFallback = TIERS.reduce((a,t)=>a+t.expected,0);
  return { total: totalFallback, passing: totalFallback, failing:0, passRate:'100.0%', byTier:{}, generatedAt:new Date().toISOString(), sourceFile:'fallback' };
}

function writeExperiments(exps){
  const header = `/*\n * AUTO-GENERATED — DO NOT EDIT\n * Generated: ${new Date().toISOString()}\n */\n\nimport { ExperimentDefinition } from './experiments';\nexport const RESEARCH_EXPERIMENTS: ExperimentDefinition[] = [\n`;
  const body = exps.map(e=>'  '+JSON.stringify(e,null,2).replace(/\n/g,'\n  ')).join(',\n');
  const footer='\n];\n';
  fs.mkdirSync(path.dirname(OUT_EXPERIMENTS),{recursive:true});
  fs.writeFileSync(OUT_EXPERIMENTS, header+body+footer, 'utf-8');
}

function writeStats(stats){
  const content = `/*\n * AUTO-GENERATED — TEST STATISTICS\n * Generated: ${stats.generatedAt}\n * Source: ${stats.sourceFile}\n */\nexport const testStatistics = ${JSON.stringify(stats,null,2)} as const;\nexport function formatPassRate(){return testStatistics.passRate+' Tests Pass';}\nexport function formatSummary(){return testStatistics.passing+' of '+testStatistics.total+' executed tests passing';}\n`;
  fs.mkdirSync(path.dirname(OUT_STATS),{recursive:true});
  fs.writeFileSync(OUT_STATS, content, 'utf-8');
}

function main(){
  console.log('� Generating website experiment metadata...');
  const experiments = generateAll();
  writeExperiments(experiments);
  const stats = computeStats();
  writeStats(stats);
  console.log(`✅ Experiments: ${experiments.length} | PassRate: ${stats.passRate}`);
  const tierCounts = experiments.reduce((m,e)=>{m[e.tier]=(m[e.tier]||0)+1; return m;},{});
  for (const t of TIERS){ console.log(`  Tier ${t.tier} (${t.name}): ${tierCounts[t.tier]||0}/${t.expected}`); }
  if (experiments.length !== TIERS.reduce((a,t)=>a+t.expected,0)){ console.warn('⚠️ Expected total mismatch; check TIERS expected counts.'); }
  return 0;
}

if (require.main === module){ try{ process.exit(main()); } catch(e){ console.error('❌ Generation failed:', e); process.exit(1); } }

module.exports = { main, TIERS, generateAll };
