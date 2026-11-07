import { validateUrl } from './urlValidation'
import { useState, type FormEvent } from 'react'
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Globe,
  LockKeyhole,
  Search,
  Shield,
  ShieldCheck,
  TriangleAlert,
} from 'lucide-react'

type AnalysisResult = {
  status: string
  normalized: boolean
  findings: string[]
  ml_analysis: {
    classification: 'flagged' | 'not_flagged'
    phishing_class_score: number
    threshold: number
    model: string
  }
  disclaimer: string
}

function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [message, setMessage] = useState('')

  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const validationError = validateUrl(url)

    if (validationError) {
      setMessage(validationError)
      return
    }

    setLoading(true)
    setResult(null)
    setMessage('Analyzing URL...')

    try {
      const response = await fetch('/api/v1/analyze/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })

      if (!response.ok) {
        const errorBody: unknown = await response.json().catch(() => null)

        let detail = `Analysis failed (HTTP ${response.status}).`

        if (
          typeof errorBody === 'object' &&
          errorBody !== null &&
          'detail' in errorBody &&
          typeof errorBody.detail === 'string'
        ) {
          detail = errorBody.detail
        }

        throw new Error(detail)
      }

      const result: AnalysisResult = await response.json()
      setResult(result)
      setMessage(
        result.ml_analysis.classification === 'flagged'
          ? 'Analysis complete: Potential threat detected. Review the findings below.'
          : 'Analysis complete: Not flagged by the model. This does not guarantee safety.'
      )
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#f4f7fb] text-slate-900">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="border-b border-slate-200 bg-white p-6 lg:w-64 lg:border-r lg:border-b-0">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-sky-100 p-3 text-sky-600">
              <ShieldCheck size={25} />
            </div>
            <div>
              <h1 className="font-bold tracking-tight">Guardian AI</h1>
              <p className="text-xs text-slate-600">Cybersecurity Platform</p>
            </div>
          </div>

          <nav aria-label="Main navigation" className="mt-10 space-y-2">
            <div className="flex items-center gap-3 rounded-xl bg-sky-50 px-4 py-3 text-sm font-medium text-sky-700">
              <Activity size={18} />
              Dashboard
            </div>
            <div className="flex items-center gap-3 px-4 py-3 text-sm text-slate-600">
              <Search size={18} />
              URL Analyzer
            </div>
            <div className="flex items-center gap-3 px-4 py-3 text-sm text-slate-600">
              <Shield size={18} />
              Threat Intelligence
            </div>
          </nav>

          <div className="mt-12 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm font-medium">Research-driven detection</p>
            <p className="mt-2 text-xs leading-5 text-slate-600">
              Machine learning and rule-based URL analysis.
            </p>
          </div>
        </aside>

        <main className="mx-auto w-full max-w-6xl flex-1 p-6 md:p-10">
          <header className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-sky-600">SECURITY WORKSPACE</p>
              <h2 className="mt-2 text-3xl font-bold tracking-tight">
                Threat Detection Dashboard
              </h2>
              <p className="mt-2 text-sm text-slate-600">
                Analyze suspicious URLs with our machine learning engine.
              </p>
            </div>
            <span className="flex items-center gap-2 rounded-full border border-slate-200 px-4 py-2 text-xs text-slate-700">
              <span className="h-2 w-2 rounded-full bg-amber-400" />
              {result ? 'API connected' : 'API not yet verified in this session'}
            </span>
          </header>

          <section className="mt-10 grid gap-4 md:grid-cols-3">
            {[
              { label: 'Detection Engine', value: 'Random Forest', icon: ShieldCheck },
              { label: 'Analysis Methods', value: 'ML + Rules', icon: Activity },
              { label: 'Current Session', value: result ? 'Scan completed' : 'No scans yet', icon: Globe },
            ].map((item) => {
              const Icon = item.icon
              return (
                <div key={item.label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md">
                  <Icon className="mb-4 text-sky-600" size={22} />
                  <p className="text-xs text-slate-600">{item.label}</p>
                  <p className="mt-2 text-lg font-semibold">{item.value}</p>
                </div>
              )
            })}
          </section>

          <section className="mt-8 rounded-2xl border border-slate-200 bg-white shadow-sm p-6 md:p-8">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-sky-50 p-3 text-sky-600">
                <Search size={22} />
              </div>
              <div>
                <h3 className="text-xl font-semibold">URL Threat Scanner</h3>
                <p className="mt-1 text-sm text-slate-600">
                  Enter a URL to run machine learning and rule-based analysis.
                </p>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="mt-7 flex flex-col gap-3 sm:flex-row">
              <label htmlFor="url-input" className="sr-only">URL to analyze</label>
              <input
                id="url-input"
                type="text"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com"
                maxLength={4096}
                className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-5 py-4 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-sky-500 focus:ring-2 focus:ring-sky-100"
              />
              <button
                type="submit"
                disabled={loading}
                className="flex items-center justify-center gap-2 rounded-xl bg-sky-600 px-6 py-4 font-semibold text-white transition hover:bg-sky-700"
              >
                {loading ? 'Analyzing...' : 'Analyze URL'} <ArrowRight size={18} />
              </button>
            </form>

            {message && (
              <p role="status" className="mt-4 text-sm text-amber-700">
                {message}
              </p>
            )}

            <p className="mt-4 flex items-center gap-2 text-xs text-slate-500">
              <LockKeyhole size={14} />
              Analysis requests are processed by the FastAPI backend.
            </p>
          </section>

          <section className="mt-8 grid gap-5 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
              <div className="flex items-center gap-2 text-slate-700">
                <CheckCircle2 size={20} className="text-sky-600" />
                <h3 className="font-semibold">ML Analysis</h3>
              </div>
              {result ? (
                <div className="mt-7 space-y-5">
                  <div className={`rounded-xl border p-5 ${
                    result.ml_analysis.classification === 'flagged'
                      ? 'border-amber-200 bg-amber-50'
                      : 'border-sky-200 bg-sky-50'
                  }`}>
                    <p className="text-xs text-slate-600">Model classification</p>
                    <p className="mt-2 text-xl font-bold">
                      {result.ml_analysis.classification === 'flagged'
                        ? 'Potential threat flagged'
                        : 'Not flagged by model'}
                    </p>
                  </div>

                  <div>
                    <div className="flex justify-between gap-3 text-sm">
                      <span className="text-slate-600">Phishing class score</span>
                      <span className="font-semibold">
                        {result.ml_analysis.phishing_class_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          result.ml_analysis.classification === 'flagged'
                            ? 'bg-amber-500'
                            : 'bg-sky-500'
                        }`}
                        style={{
                          width: `${Math.min(100, Math.max(0,
                            result.ml_analysis.phishing_class_score * 100))}%`,
                        }}
                      />
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      Decision threshold: {result.ml_analysis.threshold.toFixed(2)}
                    </p>
                  </div>

                  <p className="text-xs text-slate-600">
                    Model: {result.ml_analysis.model}
                  </p>
                  <p className="text-xs leading-5 text-slate-500">
                    This score is not a calibrated real-world phishing probability.
                  </p>
                </div>
              ) : (
                <div className="mt-8 rounded-xl border border-dashed border-slate-200 p-8 text-center">
                  <Shield size={35} className="mx-auto text-slate-500" />
                  <p className="mt-4 font-medium">No analysis available</p>
                  <p className="mt-2 text-sm text-slate-600">
                    Enter a URL and run an analysis to view results.
                  </p>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
              <div className="flex items-center gap-2 text-slate-700">
                <TriangleAlert size={20} className="text-amber-400" />
                <h3 className="font-semibold">Security Findings</h3>
              </div>
              {result ? (
                <div className="mt-7">
                  <p className="text-sm text-slate-600">
                    {result.findings.length} rule-based indicator(s) detected
                  </p>

                  {result.findings.length > 0 ? (
                    <ul className="mt-4 space-y-3">
                      {result.findings.map((finding, index) => (
                        <li
                          key={`${index}-${finding}`}
                          className="flex gap-3 rounded-xl border border-amber-200 border-l-4 border-l-amber-500 bg-amber-50 p-4 shadow-sm"
                        >
                          <TriangleAlert
                            size={18}
                            className="mt-0.5 shrink-0 text-amber-400"
                          />
                          <span className="text-sm text-slate-800">
                            {finding}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-5">
                      <CheckCircle2 size={24} className="text-sky-600" />
                      <p className="mt-3 font-medium">
                        No rule-based indicators detected
                      </p>
                      <p className="mt-2 text-sm text-slate-600">
                        None of the implemented URL rules were triggered.
                        This does not guarantee the URL is safe.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-8 rounded-xl border border-dashed border-slate-200 p-8 text-center">
                  <Activity size={35} className="mx-auto text-slate-500" />
                  <p className="mt-4 font-medium">Awaiting scan</p>
                  <p className="mt-2 text-sm text-slate-600">
                    Rule-based indicators will appear here.
                  </p>
                </div>
              )}
            </div>
          </section>

          <p className="mt-8 text-xs leading-5 text-slate-500">
            Detection results are advisory. ML scores are not calibrated real-world
            phishing probabilities. A URL not flagged by the model is not necessarily safe.
          </p>
        </main>
      </div>
    </div>
  )
}

export default App
