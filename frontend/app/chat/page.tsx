'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/lib/hooks/useAuth'
import { getSupabaseBrowserClient } from '@/lib/supabase/client'
import ChatInterface from '@/components/ChatInterface'
import Sidebar from '@/components/Sidebar'
import { StateSummary } from '@/types'
import { Session } from '@/lib/supabase/types'
import { Session as APISession } from '@/lib/api-authenticated'

export default function ChatPage() {
  const { user, profile, signOut } = useAuth()
  const [currentSession, setCurrentSession] = useState<Session | null>(null)
  const [state, setState] = useState<StateSummary | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const supabase = getSupabaseBrowserClient()

  useEffect(() => {
    if (user) {
      // Only initialize if we don't have a current session
      // This prevents switching back to recent session when tab is refocused
      if (!currentSession) {
        initializeSession()
      } else {
        setIsLoading(false)
      }
    } else {
      // User logged out - reset state
      setCurrentSession(null)
      setState(null)
      setIsLoading(false)
    }
  }, [user]) // Only depend on user, not currentSession

  const initializeSession = async () => {
    if (!user) return // Safety check
    
    try {
      // Try to get most recent session
      const { data: sessions, error: fetchError } = await supabase
        .from('sessions')
        .select('*')
        .eq('user_id', user.id)
        .order('last_activity', { ascending: false })
        .limit(1)

      if (fetchError) throw fetchError

      let session = sessions?.[0]

      // Create new session if none exists
      if (!session) {
        const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const { data: newSession, error: createError } = await supabase
          .from('sessions')
          .insert({
            user_id: user.id,
            session_id: sessionId,
            interaction_count: 0,
            current_topic: null,
            stated_goal: null,
            stated_level: null,
            performance_trend: null,
            understanding_scores: '[]',
            source_files: '[]', // Initialize empty array for RAG source files
          } as any)
          .select()
          .single()

        if (createError) throw createError
        session = newSession
      }

      setCurrentSession(session)
    } catch (error) {
      // Silently handle initialization errors
    } finally {
      setIsLoading(false)
    }
  }

  const handleStateUpdate = (newState: StateSummary) => {
    setState(newState)
  }

  const handleNewSession = async () => {
    if (!user) {
      return
    }
    
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    try {
      // Create session with only required fields that exist in the database
      // Note: difficulty, mastered_concepts, learning_style, onboarding_complete are NOT in sessions table
      // (they're in profiles table). source_files IS in sessions table (for RAG traceability)
      const { data: newSession, error } = await supabase
        .from('sessions')
        .insert({
          user_id: user.id,
          session_id: sessionId,
          interaction_count: 0,
          current_topic: null,
          stated_goal: null,
          stated_level: null,
          performance_trend: null,
          understanding_scores: '[]',
          source_files: '[]', // Initialize empty array for RAG source files
        } as any)
        .select()
        .single()

      if (error) {
        alert(`Failed to create new session: ${error.message}`)
        throw error
      }
      
      if (!newSession) {
        alert('Failed to create new session: No data returned')
        return
      }
      
      // Set the new session - this will trigger ChatInterface to load and show welcome message
      setCurrentSession(newSession)
      setState(null) // Reset state for new session
      setIsLoading(false) // Make sure loading is false so ChatInterface renders
      
    } catch (error: any) {
      alert(`Error creating new session: ${error?.message || 'Unknown error'}`)
    }
  }

  const handleSessionSelect = async (session: APISession) => {
    // Convert API session to Supabase session type (they have the same structure)
    setCurrentSession(session as Session)
    setState(null) // Reset state, will be loaded by ChatInterface
  }

  // Redirect to login if user is not authenticated
  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-gray-500 mb-4">Please log in to continue</p>
          <a href="/login" className="text-indigo-600 hover:text-indigo-700">
            Go to Login
          </a>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar - Fixed position */}
      <div className="flex-shrink-0">
        <Sidebar
          state={state}
          sessionId={currentSession?.session_id || null}
          profile={profile}
          onSignOut={signOut}
          onNewSession={handleNewSession}
          onSessionSelect={handleSessionSelect}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {currentSession ? (
          <ChatInterface
            sessionId={currentSession.session_id}
            userId={user.id}
            onStateUpdate={handleStateUpdate}
            currentState={state}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">Failed to initialize session</p>
          </div>
        )}
      </div>
    </div>
  )
}
