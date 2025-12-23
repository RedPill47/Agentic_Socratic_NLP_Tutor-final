export interface RAGChunk {
  content: string
  source_file: string
  page_number?: number | string
  slide_title?: string
  topic?: string
  difficulty?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  metadata?: {
    learning_style_detected?: string
    adaptation_event?: string
    evaluation?: {
      correctness: 'correct' | 'partially_correct' | 'incorrect'
      depth: 'deep' | 'moderate' | 'surface'
    }
    rag_sources?: RAGChunk[]
  }
}

export interface StateSummary {
  session_id: string
  student_id: string
  current_topic: string | null
  difficulty: string
  interaction_count: number
  learning_style: {
    primary_style: string
    confidence: number
    indicators: string[]
  } | null
  performance: PerformanceMetrics | null
  completed_topics: string[]
  detected_gaps: string[]
  prerequisites_to_revisit: string[]
  source_files: string[]  // RAG source files used
  last_adaptation_event: string | null
}

export interface PerformanceMetrics {
  correctness_score: number
  depth_score: number
  consecutive_correct: number
  consecutive_incorrect: number
  trend: 'improving' | 'declining' | 'stable'
}

export interface LearningStyle {
  primary_style: 'visual' | 'auditory' | 'kinesthetic' | 'reading'
  confidence: number
  indicators: string[]
  last_updated: string
}

