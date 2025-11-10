/*
 * © 2025 Emergent Physics Lab. All rights reserved.
 * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
 */

'use client';

import { useParams } from 'next/navigation';
import { getExperimentById } from '@/data/experiments';
import ExperimentTemplate from '@/components/experiments/ExperimentTemplate';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import Link from 'next/link';

export default function ExperimentPage() {
  const params = useParams();
  const id = params?.id as string;
  
  const experiment = getExperimentById(id);
  
  if (!experiment) {
    return (
      <div className="min-h-screen flex flex-col bg-space-dark">
        <Header />
        <main className="flex-1 pt-20 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-accent-chi mb-4">Experiment Not Found</h1>
            <p className="text-text-secondary mb-8">
              The experiment "{id}" could not be found.
            </p>
            <Link href="/research" className="button-primary">
              ← Back to Research
            </Link>
          </div>
        </main>
        <Footer />
      </div>
    );
  }
  
  return (
    <div className="min-h-screen flex flex-col bg-space-dark">
      <Header />
      <main className="flex-1 pt-20">
        <ExperimentTemplate experiment={experiment} />
      </main>
      <Footer />
    </div>
  );
}
