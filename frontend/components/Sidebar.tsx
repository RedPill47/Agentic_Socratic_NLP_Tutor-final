'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { StateSummary } from '@/types'
import { Profile } from '@/lib/supabase/types'
import { BookOpen, TrendingUp, Target, AlertCircle, CheckCircle2, LogOut, Plus, User, MessageSquare, Trash2, Loader2, FileText, Eye } from 'lucide-react'
import Link from 'next/link'
import PerformanceChart from './PerformanceChart'
import { AuthenticatedAPI, Session } from '@/lib/api-authenticated'
import RAGMaterialsModal from './RAGMaterialsModal'

interface SidebarProps {
  state: StateSummary | null
  sessionId: string | null
  profile: Profile | null
  onSignOut: () => void
  onNewSession?: () => void
  onSessionSelect?: (session: Session) => void
}

export default function Sidebar({ state, sessionId, profile, onSignOut, onNewSession, onSessionSelect }: SidebarProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null)
  const [showRAGModal, setShowRAGModal] = useState(false)

  const styleEmoji: Record<string, string> = {
    visual: '👁️',
    auditory: '👂',
    kinesthetic: '✋',
    reading: '📖',
  }

  const loadSessions = useCallback(async () => {
    try {
      setLoadingSessions(true)
      const fetchedSessions = await AuthenticatedAPI.getSessions()
      setSessions(fetchedSessions)
    } catch (error) {
      // Silently handle session loading errors
    } finally {
      setLoadingSessions(false)
    }
  }, [])

  // Fetch sessions on mount and when sessionId changes
  useEffect(() => {
    loadSessions()
  }, [sessionId, loadSessions])

  const handleSessionClick = (session: Session) => {
    if (onSessionSelect && session.session_id !== sessionId) {
      onSessionSelect(session)
    }
  }

  const handleDeleteSession = async (e: React.MouseEvent, session: Session) => {
    e.stopPropagation() // Prevent triggering session selection
    
    if (!confirm(`Are you sure you want to delete this chat? This action cannot be undone.`)) {
      return
    }

    try {
      setDeletingSessionId(session.session_id)
      await AuthenticatedAPI.deleteSession(session.session_id)
      
      // Remove from local state
      setSessions(prev => prev.filter(s => s.session_id !== session.session_id))
      
      // If we deleted the current session, create a new one
      if (session.session_id === sessionId && onNewSession) {
        onNewSession()
      }
    } catch (error: any) {
      alert('Failed to delete session. Please try again.')
    } finally {
      setDeletingSessionId(null)
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-screen overflow-hidden">
      {/* User Profile Header */}
      <div className="p-6 border-b border-gray-200 flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 bg-indigo-100 rounded-full flex items-center justify-center">
              <User className="h-5 w-5 text-indigo-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900">
                {profile?.full_name || 'Student'}
              </p>
              <p className="text-xs text-gray-500 capitalize">{profile?.role}</p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          {onNewSession && (
            <button
              onClick={onNewSession}
              className="flex-1 flex items-center justify-center gap-1 px-3 py-2 text-sm bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 transition-colors"
            >
              <Plus className="h-4 w-4" />
              New Session
            </button>
          )}
          {profile?.role === 'teacher' && (
            <Link
              href="/teacher"
              className="flex-1 flex items-center justify-center gap-1 px-3 py-2 text-sm bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition-colors"
            >
              📚 Dashboard
            </Link>
          )}
        </div>

        <button
          onClick={onSignOut}
          className="mt-2 w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
        >
          <LogOut className="h-4 w-4" />
          Sign Out
        </button>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {/* Chat History Section */}
        <div className="p-4 border-b border-gray-200 flex-shrink-0">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
            <MessageSquare className="h-4 w-4 mr-2" />
            Chat History
          </h3>
          
          {loadingSessions ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
            </div>
          ) : sessions.length === 0 ? (
            <p className="text-xs text-gray-500 text-center py-2">No chat history yet</p>
          ) : (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {sessions.map((session) => {
                const isActive = session.session_id === sessionId
                return (
                  <div
                    key={session.session_id}
                    onClick={() => handleSessionClick(session)}
                    className={`group relative p-2 rounded-lg cursor-pointer transition-colors ${
                      isActive
                        ? 'bg-indigo-50 border border-indigo-200'
                        : 'hover:bg-gray-50 border border-transparent'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <div className={`h-2 w-2 rounded-full flex-shrink-0 ${
                            isActive ? 'bg-indigo-600' : 'bg-gray-300'
                          }`} />
                          <p className={`text-xs font-medium truncate ${
                            isActive ? 'text-indigo-900' : 'text-gray-900'
                          }`}>
                            {session.current_topic || 'New Chat'}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-500 ml-4">
                          <span>{formatDate(session.last_activity)}</span>
                          <span>•</span>
                          <span>{session.interaction_count} messages</span>
                        </div>
                      </div>
                      <button
                        onClick={(e) => handleDeleteSession(e, session)}
                        disabled={deletingSessionId === session.session_id}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity flex-shrink-0"
                        title="Delete chat"
                      >
                        {deletingSessionId === session.session_id ? (
                          <Loader2 className="h-3 w-3 animate-spin text-red-600" />
                        ) : (
                          <Trash2 className="h-3 w-3 text-red-600" />
                        )}
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Session State Section */}
        <div className="flex-1 p-6 space-y-6 overflow-y-auto min-h-0">
          {!state ? (
            <div className="text-center text-gray-500 mt-8">
              <p>Start a conversation to see your learning progress</p>
            </div>
          ) : (
            <>
              {/* Session Info */}
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Current Session</h3>
                <div className="text-xs text-gray-500 space-y-1">
                  <p>Interactions: {state.interaction_count}</p>
                </div>
              </div>

        <div className="border-t border-gray-200 pt-6">
          {/* Learning State */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <BookOpen className="h-4 w-4 mr-2" />
              Learning State
            </h3>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-500">Topic:</span>{' '}
                <span className="font-medium text-gray-900">
                  {state.current_topic || 'Not selected'}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Difficulty:</span>{' '}
                <span className="font-medium text-gray-900 capitalize">
                  {state.difficulty}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Completed:</span>{' '}
                <span className="font-medium text-gray-900">
                  {state.completed_topics.length} topics
                </span>
              </div>
            </div>
          </div>

          {/* Learning Style */}
          {state.learning_style && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <Target className="h-4 w-4 mr-2" />
                Learning Style
              </h3>
              <div className="bg-primary-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-2xl">
                    {styleEmoji[state.learning_style.primary_style] || '📖'}
                  </span>
                  <div>
                    <div className="font-medium text-gray-900 capitalize">
                      {state.learning_style.primary_style}
                    </div>
                    <div className="text-xs text-gray-600">
                      {Math.round(state.learning_style.confidence * 100)}% confidence
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Source Files */}
          {state.source_files && state.source_files.length > 0 && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center">
                  <FileText className="h-4 w-4 mr-2" />
                  Source Files
                </h3>
                <button
                  onClick={() => setShowRAGModal(true)}
                  className="text-xs flex items-center gap-1 px-2 py-1 text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded transition-colors"
                  title="View RAG materials"
                >
                  <Eye className="h-3 w-3" />
                  View Materials
                </button>
              </div>
              <div className="space-y-2">
                {state.source_files.map((file, index) => (
                  <div
                    key={index}
                    className="text-xs bg-blue-50 border border-blue-200 rounded p-2 text-gray-700"
                  >
                    <span className="font-medium">📄</span>{' '}
                    <span className="truncate block">{file}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {state.performance && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2" />
                Performance
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Correctness</span>
                    <span className="font-medium text-gray-900">
                      {Math.round(state.performance.correctness_score * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{
                        width: `${state.performance.correctness_score * 100}%`,
                      }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Depth</span>
                    <span className="font-medium text-gray-900">
                      {Math.round(state.performance.depth_score * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{
                        width: `${state.performance.depth_score * 100}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Trend:</span>
                  <span
                    className={`font-medium ${
                      state.performance.trend === 'improving'
                        ? 'text-green-600'
                        : state.performance.trend === 'declining'
                        ? 'text-red-600'
                        : 'text-gray-600'
                    }`}
                  >
                    {state.performance.trend}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-center p-2 bg-green-50 rounded">
                    <div className="font-semibold text-green-700">
                      {state.performance.consecutive_correct}
                    </div>
                    <div className="text-green-600">Correct</div>
                  </div>
                  <div className="text-center p-2 bg-red-50 rounded">
                    <div className="font-semibold text-red-700">
                      {state.performance.consecutive_incorrect}
                    </div>
                    <div className="text-red-600">Incorrect</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Knowledge Gaps */}
          {state.detected_gaps.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                <AlertCircle className="h-4 w-4 mr-2 text-yellow-500" />
                Knowledge Gaps
              </h3>
              <ul className="space-y-1 text-xs text-gray-600">
                {state.detected_gaps.slice(0, 3).map((gap, index) => (
                  <li key={index} className="flex items-start">
                    <span className="mr-2">•</span>
                    <span>{gap}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Prerequisites */}
          {state.prerequisites_to_revisit.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                <CheckCircle2 className="h-4 w-4 mr-2 text-blue-500" />
                Prerequisites to Revisit
              </h3>
              <ul className="space-y-1 text-xs text-gray-600">
                {state.prerequisites_to_revisit.slice(0, 3).map((prereq, index) => (
                  <li key={index} className="flex items-start">
                    <span className="mr-2">•</span>
                    <span>{prereq}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Last Adaptation */}
          {state.last_adaptation_event && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                Last Adaptation
              </h3>
              <div className="text-xs text-gray-600 bg-blue-50 rounded p-2">
                {state.last_adaptation_event.replace('_', ' ').replace(/\b\w/g, (l) =>
                  l.toUpperCase()
                )}
              </div>
            </div>
          )}
        </div>
            </>
          )}
        </div>
      </div>

      {/* RAG Materials Modal */}
      <RAGMaterialsModal
        isOpen={showRAGModal}
        onClose={() => setShowRAGModal(false)}
        sessionId={sessionId}
        sourceFiles={state?.source_files || []}
        currentTopic={state?.current_topic || null}
      />
    </div>
  )
}

