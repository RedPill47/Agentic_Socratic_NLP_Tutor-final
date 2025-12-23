import { ChatMessage, StateSummary, PerformanceMetrics } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ChatRequest {
  content: string
  student_id?: string
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
  }
}

export class API {
  static async chat(message: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  }

  static async chatStream(
    message: ChatRequest,
    onToken: (token: string) => void,
    onMetadata: (metadata: any) => void,
    onError: (error: Error) => void,
    onState?: (state: string, message: string) => void
  ): Promise<void> {
    const response = await fetch(`${API_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
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

            if (data.type === 'token') {
              onToken(data.content)
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

  static async getState(sessionId: string): Promise<StateSummary> {
    const response = await fetch(`${API_URL}/api/state/${sessionId}`)

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  }

  static async updateState(
    sessionId: string,
    updates: {
      topic?: string
      difficulty?: string
      mark_topic_completed?: string
    }
  ): Promise<StateSummary> {
    const response = await fetch(`${API_URL}/api/state/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    const data = await response.json()
    return data.state
  }

  static async getMetrics(sessionId: string): Promise<{
    performance: PerformanceMetrics | null
    learning_style: any
    interaction_count: number
    completed_topics: string[]
    detected_gaps: string[]
    prerequisites_to_revisit: string[]
  }> {
    const response = await fetch(`${API_URL}/api/metrics/${sessionId}`)

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  }

  static async resetSession(sessionId: string): Promise<void> {
    await fetch(`${API_URL}/api/state/${sessionId}`, {
      method: 'DELETE',
    })
  }
}

