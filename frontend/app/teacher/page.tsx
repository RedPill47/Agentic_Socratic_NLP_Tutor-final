'use client'

import { useState, useEffect, useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/lib/hooks/useAuth'
import { getSupabaseBrowserClient } from '@/lib/supabase/client'
import {
  Upload,
  FileText,
  Loader2,
  CheckCircle2,
  XCircle,
  Users,
  BookOpen,
  FileCheck,
  Database,
  Brain,
  LogOut,
  AlertTriangle,
} from 'lucide-react'
import { Profile, Session } from '@/lib/supabase/types'

interface SupabaseSession extends Session {
  user_id: string
}

interface StudentRow {
  id: string
  name: string
  topic: string
  style: string
  mastery: number | null
  gap: string
}

export default function TeacherDashboard() {
  const router = useRouter()
  const { profile, signOut } = useAuth()
  const supabase = getSupabaseBrowserClient()

  const [uploading, setUploading] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const [loadingData, setLoadingData] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const [sessions, setSessions] = useState<SupabaseSession[]>([])
  const [documents, setDocuments] = useState<any[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [masAnalyses, setMasAnalyses] = useState<any[]>([])

  useEffect(() => {
    if (!profile) return
    const fetchData = async () => {
      setLoadingData(true)
      setError(null)
      try {
        // Fetch data from actual tables that are used
        const [sessionsRes, docsRes, profilesRes, masRes] = await Promise.all([
          supabase.from('sessions').select('*').order('last_activity', { ascending: false }),
          supabase.from('documents').select('*').order('uploaded_at', { ascending: false }),
          supabase.from('profiles').select('*').eq('role', 'student'),
          supabase.from('mas_analysis').select('*').order('created_at', { ascending: false }),
        ])

        if (sessionsRes.error) {
          throw new Error(`Failed to load sessions: ${sessionsRes.error.message}`)
        }
        if (profilesRes.error) {
          throw new Error(`Failed to load profiles: ${profilesRes.error.message}`)
        }
        // documents and mas_analysis errors are non-critical, silently ignore

        setSessions(sessionsRes.data || [])
        setDocuments(docsRes.data || [])
        setProfiles(profilesRes.data || [])
        setMasAnalyses(masRes.data || [])
      } catch (err: any) {
        setError(err.message || 'Failed to load dashboard data. Make sure RLS policies allow teacher access.')
      } finally {
        setLoadingData(false)
      }
    }

    fetchData()
  }, [profile, supabase])

  const handleSignOut = async () => {
    setSigningOut(true)
    setMessage(null)
    try {
      await signOut()
      router.push('/login')
    } catch (error: any) {
      setMessage({
        type: 'error',
        text: error.message || 'Failed to sign out',
      })
      setSigningOut(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!profile) {
      setMessage({ type: 'error', text: 'You must be signed in to upload documents.' })
      return
    }

    if (!file.name.endsWith('.pdf')) {
      setMessage({ type: 'error', text: 'Please upload a PDF file' })
      return
    }

    setUploading(true)
    setMessage(null)

    try {
      const filePath = `documents/${Date.now()}_${file.name}`
      const { error: uploadError } = await supabase.storage.from('documents').upload(filePath, file)

      if (uploadError) throw uploadError

      const { error: dbError } = await supabase.from('documents').insert({
        uploaded_by: profile!.id,
        filename: file.name,
        file_path: filePath,
        status: 'pending',
      } as any)

      if (dbError) throw dbError

      setMessage({
        type: 'success',
        text: `${file.name} uploaded successfully! Processing will begin shortly.`,
      })

      // Refresh documents list
      const { data } = await supabase.from('documents').select('*').order('uploaded_at', { ascending: false })
      if (data) setDocuments(data)

      e.target.value = ''
    } catch (error: any) {
      setMessage({
        type: 'error',
        text: error.message || 'Failed to upload file',
      })
    } finally {
      setUploading(false)
    }
  }

  // Calculate analytics
  const analyticsCards = useMemo(() => {
    const studentCount = profiles.length
    const sessionCount = sessions.length
    const docCount = documents.length
    const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunks_created ?? 0), 0)

    return [
      {
        label: 'Total Students',
        value: studentCount,
        sub: `${profiles.filter((p) => p.last_activity).length} active`,
        icon: Users,
        tone: 'blue' as const,
      },
      {
        label: 'Total Sessions',
        value: sessionCount,
        sub: `${new Set(sessions.map((s) => s.user_id)).size} unique users`,
        icon: BookOpen,
        tone: 'green' as const,
      },
      {
        label: 'Documents',
        value: docCount,
        sub: `${documents.filter((d) => d.status === 'completed').length} processed`,
        icon: FileCheck,
        tone: 'purple' as const,
      },
      {
        label: 'Total Chunks',
        value: totalChunks,
        sub: 'Vector store entries',
        icon: Database,
        tone: 'orange' as const,
      },
    ]
  }, [profiles, sessions, documents])

  // Calculate student rows
  const studentRows: StudentRow[] = useMemo(() => {
    // Get most recent session per user
    const sessionsByUser = new Map<string, SupabaseSession>()
    sessions.forEach((s) => {
      const existing = sessionsByUser.get(s.user_id)
      if (!existing || new Date(s.last_activity) > new Date(existing.last_activity)) {
        sessionsByUser.set(s.user_id, s)
      }
    })

    // Get latest MAS analysis per user for performance data
    const masByUser = new Map<string, any>()
    masAnalyses.forEach((analysis) => {
      const existing = masByUser.get(analysis.user_id)
      if (!existing || new Date(analysis.created_at) > new Date(existing.created_at)) {
        masByUser.set(analysis.user_id, analysis)
      }
    })

    // Calculate mastery and gaps from sessions and MAS analysis
    const userStats = new Map<string, { mastery: number | null; gap: string }>()

    profiles.forEach((p) => {
      if (p.role !== 'student') return

      // Get all sessions for this user
      const userSessions = sessions.filter((s) => s.user_id === p.id)

      // Calculate mastery from understanding_scores across all sessions
      let allScores: number[] = []
      userSessions.forEach((session) => {
        try {
          const scores = session.understanding_scores ? JSON.parse(session.understanding_scores) : []
          if (Array.isArray(scores)) {
            allScores.push(...scores)
          }
        } catch (e) {
          // Ignore parse errors
        }
      })

      // Also check MAS analysis for performance data
      const masAnalysis = masByUser.get(p.id)
      if (masAnalysis?.performance_assessment) {
        const perf = masAnalysis.performance_assessment
        if (typeof perf === 'object' && perf.correctness_score !== null) {
          allScores.push(perf.correctness_score)
        }
      }

      const mastery = allScores.length > 0
        ? Math.round((allScores.reduce((a, b) => a + b, 0) / allScores.length) * 100)
        : null

      // Find knowledge gaps from MAS analysis
      let gap = 'None'
      if (masAnalysis?.knowledge_gaps && Array.isArray(masAnalysis.knowledge_gaps) && masAnalysis.knowledge_gaps.length > 0) {
        gap = masAnalysis.knowledge_gaps[0]
      }

      userStats.set(p.id, { mastery, gap })
    })

    return profiles
      .filter((p) => p.role === 'student')
      .map((p) => {
        const session = sessionsByUser.get(p.id)
        const stats = userStats.get(p.id) || { mastery: null, gap: 'None' }

        // Get learning style from profiles table (user-level)
        const style = p.learning_style || '—'

        // Get current topic from most recent session
        const topic = session?.current_topic || '—'

        return {
          id: p.id,
          name: p.full_name || 'Student',
          topic,
          style: style !== '—' ? style.charAt(0).toUpperCase() + style.slice(1) : '—',
          mastery: stats.mastery,
          gap: stats.gap,
        }
      })
  }, [profiles, sessions, masAnalyses])

  const toneClasses = {
    blue: 'border-blue-200 bg-blue-50 text-blue-600',
    green: 'border-green-200 bg-green-50 text-green-600',
    purple: 'border-purple-200 bg-purple-50 text-purple-600',
    orange: 'border-orange-200 bg-orange-50 text-orange-600',
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Agentic Socratic NLP Tutor</p>
            <h1 className="text-2xl font-semibold text-slate-900">Teacher Dashboard</h1>
            <p className="text-sm text-slate-500">Monitor student progress, system health, and knowledge ingestion.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleSignOut}
              disabled={signingOut}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-red-600 transition hover:border-red-200 hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <LogOut className="h-4 w-4" />
              {signingOut ? 'Signing out...' : 'Sign Out'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {message && (
          <div
            className={`mb-6 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${
              message.type === 'success'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                : 'border-amber-200 bg-amber-50 text-amber-800'
            }`}
          >
            {message.type === 'success' ? (
              <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0" />
            ) : (
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0" />
            )}
            <p>{message.text}</p>
          </div>
        )}

        {/* Data load error */}
        {error && (
          <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            {error}
          </div>
        )}

        {/* Analytics */}
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {analyticsCards.map((card) => (
            <div key={card.label} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">{card.label}</p>
                  <p className="mt-2 text-2xl font-semibold text-slate-900">{card.value}</p>
                  <p className="text-sm text-slate-500">{card.sub}</p>
                </div>
                <div className={`rounded-xl border p-2 ${toneClasses[card.tone]}`}>
                  <card.icon className="h-5 w-5" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Main content */}
        <div className="mt-8 grid gap-6 lg:grid-cols-3">
          {/* Student Progress */}
          <div className="lg:col-span-2 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Student Progress</h2>
                <p className="text-sm text-slate-500">Live signals from the tutoring agents</p>
              </div>
              <div className="flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                <Brain className="h-4 w-4" />
                Adaptive insights enabled
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                    <th className="pb-3">Student</th>
                    <th className="pb-3">Current Topic</th>
                    <th className="pb-3">Learning Style</th>
                    <th className="pb-3">Mastery</th>
                    <th className="pb-3">Last Session Gap</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {loadingData ? (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-sm text-slate-500">
                        Loading student progress...
                      </td>
                    </tr>
                  ) : studentRows.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-sm text-slate-500">
                        No student data available yet.
                      </td>
                    </tr>
                  ) : (
                    studentRows.map((s) => (
                      <tr key={s.id} className="align-middle">
                        <td className="py-3 font-semibold text-slate-900">{s.name}</td>
                        <td className="py-3 text-slate-600">{s.topic}</td>
                        <td className="py-3">
                          <span className="inline-flex items-center rounded-full bg-purple-50 px-2 py-1 text-xs font-medium text-purple-700">
                            {s.style}
                          </span>
                        </td>
                        <td className="py-3">
                          {s.mastery !== null ? (
                            <span className="font-semibold text-slate-900">{s.mastery}%</span>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </td>
                        <td className="py-3 text-slate-600">{s.gap}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Document Management */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Knowledge Base</h2>
                <p className="text-sm text-slate-500">Upload and manage documents</p>
              </div>
            </div>

            {/* Upload Area */}
            <div className="mb-6 border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-indigo-400 transition-colors">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className={`cursor-pointer ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {uploading ? (
                  <Loader2 className="h-8 w-8 text-indigo-600 mx-auto mb-2 animate-spin" />
                ) : (
                  <Upload className="h-8 w-8 text-slate-400 mx-auto mb-2" />
                )}
                <p className="text-sm text-slate-600 mb-1">
                  {uploading ? 'Uploading...' : 'Click to upload PDF'}
                </p>
                <p className="text-xs text-slate-500">PDF files only</p>
              </label>
            </div>

            {/* Documents List */}
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-900">Recent Uploads</h3>
              {documents.length === 0 ? (
                <p className="text-sm text-slate-500">No documents uploaded yet.</p>
              ) : (
                documents.slice(0, 5).map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between rounded-lg border border-slate-200 p-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-900 truncate">{doc.filename}</p>
                      <p className="text-xs text-slate-500">
                        {doc.status === 'completed' ? 'Indexed' : doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
                      </p>
                    </div>
                    {doc.chunks_created > 0 && (
                      <span className="ml-2 text-xs text-slate-600">{doc.chunks_created} chunks</span>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
