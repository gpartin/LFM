/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Dynamic research experiment page — renders 1:1 mapping to test harness
 */

import { getExperimentById } from '@/data/experiments';
import ResearchExperimentPanel from '@/components/research/ResearchExperimentPanel';
import Link from 'next/link';

interface PageProps {
  params: { id: string };
}

export default async function ResearchExperimentPage({ params }: PageProps) {
  const exp = getExperimentById(params.id);

  if (!exp) {
    return (
      <div className="min-h-screen bg-slate-950 text-white p-8">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-3xl font-bold mb-4">Experiment not found</h1>
          <p className="text-slate-300 mb-6">No experiment with id: {params.id}</p>
          <Link href="/research" className="text-indigo-400 hover:text-indigo-300 underline">Back to Research</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 text-white p-6 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6 flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl md:text-3xl font-bold">{exp.displayName}</h1>
              {exp.tier && (
                <span className="px-2 py-1 rounded bg-indigo-600/30 text-indigo-200 text-sm">Tier {exp.tier}</span>
              )}
              {exp.simulation && (
                <span className="px-2 py-1 rounded bg-slate-700 text-slate-200 text-sm">{exp.simulation}</span>
              )}
              <span className={`px-2 py-1 rounded text-sm ${exp.status === 'production' ? 'bg-green-600/30 text-green-200' : exp.status === 'beta' ? 'bg-yellow-600/30 text-yellow-200' : exp.status === 'development' ? 'bg-orange-600/30 text-orange-200' : 'bg-slate-600/30 text-slate-200'}`}>{exp.status}</span>
            </div>
            <p className="text-slate-300 max-w-3xl">{exp.tagline}</p>
          </div>
          <Link href="/research" className="text-indigo-400 hover:text-indigo-300 underline">All experiments</Link>
        </div>

        <ResearchExperimentPanel experiment={exp} />
      </div>
    </div>
  );
}
