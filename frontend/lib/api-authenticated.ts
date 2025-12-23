/**
 * Authenticated API client with JWT token support
 */

import { ChatMessage, StateSummary } from '@/types'
import { getSupabaseBrowserClient } from './supabase/client'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ChatRequest {
  content: string
  session_id?: string
}

export interface ChatResponse {
  content: string
  session_id: string
  student_id: string
  metadata?: {
    learning_style?: any
    performance?: any
    adaptation_event?: string
    state?: string
    topic?: string
  }
}

export interface Session {
  id: string
  user_id: string
  session_id: string
  current_topic: string | null
  difficulty: string
  interaction_count: number
  created_at: string
  last_activity: string
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  metadata: any
  created_at: string
}

/**
 * Get JWT token from Supabase session
 */
async function getAuthToken(): Promise<string> {
  const supabase = getSupabaseBrowserClient()
  const { data: { session } } = await supabase.auth.getSession()

  if (!session?.access_token) {
    throw new Error('Not authenticated')
  }

  return session.access_token
}

/**
 * Create headers with authentication
 */
async function getAuthHeaders(): Promise<HeadersInit> {
  const token = await getAuthToken()
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  }
}

export class AuthenticatedAPI {
  /**
   * Send chat message (non-streaming)
   */
  static async chat(message: ChatRequest): Promise<ChatResponse> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(message),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(error.detail || 'API error')
    }

    return response.json()
  }

  /**
   * Send chat message with streaming response
   */
  static async chatStream(
    message: ChatRequest,
    onToken: (token: string) => void,
    onMetadata: (metadata: any) => void,
    onError: (error: Error) => void,
    onState?: (state: string, message: string) => void
  ): Promise<void> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(message),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(error.detail || 'API error')
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) {
      throw new Error('No response body')
    }

    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))

            // Handle streaming chunks (backend sends 'chunk', legacy format uses 'token')
            if (data.type === 'chunk' || data.type === 'token') {
              if (data.content) {
                onToken(data.content)
              }
            } else if (data.type === 'done') {
              // Stream complete, send final metadata if available
              if (data.metadata) {
                onMetadata(data.metadata)
              }
            } else if (data.type === 'metadata') {
              onMetadata(data)
            } else if (data.type === 'state' && onState) {
              onState(data.state, data.message || '')
            } else if (data.type === 'status' && onState) {
              onState('processing', data.content)
            } else if (data.type === 'error') {
              onError(new Error(data.content))
            }
          } catch (e) {
            // Silently handle parsing errors
          }
        }
      }
    }
  }

  /**
   * Get all sessions for current user
   */
  static async getSessions(): Promise<Session[]> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/sessions`, {
      method: 'GET',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to fetch sessions')
    }

    const data = await response.json()
    return data.sessions || []
  }

  /**
   * Get messages for a specific session
   */
  static async getSessionMessages(sessionId: string): Promise<Message[]> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/sessions/${sessionId}/messages`, {
      method: 'GET',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to fetch messages')
    }

    const data = await response.json()
    return data.messages || []
  }

  /**
   * Delete a session
   */
  static async deleteSession(sessionId: string): Promise<void> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to delete session')
    }
  }

  /**
   * Get RAG materials for a session
   */
  static async getRAGMaterials(sessionId: string): Promise<{ chunks: any[] }> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/sessions/${sessionId}/rag-materials`, {
      method: 'GET',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to fetch RAG materials')
    }

    return response.json()
  }

  /**
   * Get current session state
   */
  static async getState(sessionId: string): Promise<StateSummary> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/state/${sessionId}`, {
      method: 'GET',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to fetch state')
    }

    const data = await response.json()
    // Map backend StateSummary to frontend StateSummary format
    return {
      session_id: data.session_id,
      student_id: '', // Not provided by backend
      current_topic: data.current_topic,
      difficulty: data.difficulty,
      interaction_count: data.interaction_count,
      learning_style: data.learning_style ? {
        primary_style: data.learning_style,
        confidence: 0.8, // Default if not provided
        indicators: []
      } : null,
      performance: null, // Not in backend StateSummary
      completed_topics: data.mastered_concepts || [],
      detected_gaps: [],
      prerequisites_to_revisit: [],
      source_files: data.source_files || [],
      last_adaptation_event: data.last_adaptation_event
    }
  }

  /**
   * Get welcome message for a new session
   */
  static async getWelcomeMessage(sessionId: string): Promise<{ content: string; session_id: string }> {
    const headers = await getAuthHeaders()

    const response = await fetch(`${API_URL}/api/sessions/${sessionId}/welcome`, {
      method: 'POST',
      headers,
    })

    if (!response.ok) {
      throw new Error('Failed to get welcome message')
    }

    return response.json()
  }
}
