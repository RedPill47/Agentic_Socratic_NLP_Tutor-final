/**
 * Database types generated from Supabase schema
 */

export type UserRole = 'student' | 'teacher'

export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced'

export type MessageRole = 'user' | 'assistant' | 'system'

export type ProgressStatus = 'not_started' | 'in_progress' | 'completed'

export type LearningStyle = 'visual' | 'auditory' | 'kinesthetic' | 'reading'

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface Profile {
  id: string
  email: string
  full_name: string | null
  role: UserRole
  created_at: string
  updated_at: string
  // User-level learning data (from add_user_level_fields migration)
  learning_style?: string | null
  learning_style_confidence?: number | null
  learning_style_updated_at?: string | null
  mastered_concepts?: string | null // JSON array as TEXT
  mastered_concepts_updated_at?: string | null
  overall_difficulty?: DifficultyLevel | null
  difficulty_updated_at?: string | null
  total_sessions?: number | null
  total_interactions?: number | null
  last_activity?: string | null
  onboarding_complete?: boolean | null
  onboarding_completed_at?: string | null
}

export interface Session {
  id: string
  user_id: string
  session_id: string
  current_topic: string | null
  difficulty: DifficultyLevel
  interaction_count: number
  created_at: string
  last_activity: string
  // Additional fields from add_session_state_fields migration
  mastered_concepts?: string | null // JSON array as TEXT
  learning_style?: string | null
  conversation_history?: string | null // JSON array as TEXT
  onboarding_complete?: boolean | null
  stated_goal?: string | null
  stated_level?: string | null
  understanding_scores?: string | null // JSON array as TEXT
  performance_trend?: string | null
  source_files?: string | null // JSON array as TEXT
  mas_analysis?: any | null // JSONB
  last_analysis_at?: string | null
  last_updated?: string | null
}

export interface Message {
  id: string
  session_id: string
  role: MessageRole
  content: string
  metadata: {
    learning_style?: any
    performance?: any
    adaptation_event?: string
    state?: string
    topic?: string
  }
  created_at: string
}

export interface Progress {
  id: string
  user_id: string
  topic: string
  status: ProgressStatus
  correctness_score: number
  depth_score: number
  consecutive_correct: number
  last_studied: string
  completed_at: string | null
}

export interface LearningStyleData {
  id: string
  user_id: string
  primary_style: LearningStyle | null
  confidence: number
  indicators: Record<string, any>
  detected_at: string
  updated_at: string
}

export interface Document {
  id: string
  uploaded_by: string
  filename: string
  file_path: string
  status: DocumentStatus
  chunks_created: number
  error_message: string | null
  uploaded_at: string
  processed_at: string | null
}

// Database type for Supabase client
export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: Profile
        Insert: Omit<Profile, 'created_at' | 'updated_at'>
        Update: Partial<Omit<Profile, 'id' | 'created_at' | 'updated_at'>>
      }
      sessions: {
        Row: Session
        Insert: Omit<Session, 'id' | 'created_at' | 'last_activity'>
        Update: Partial<Omit<Session, 'id' | 'user_id' | 'created_at'>>
      }
      messages: {
        Row: Message
        Insert: Omit<Message, 'id' | 'created_at'>
        Update: Partial<Omit<Message, 'id' | 'session_id' | 'created_at'>>
      }
      progress: {
        Row: Progress
        Insert: Omit<Progress, 'id' | 'last_studied' | 'completed_at'>
        Update: Partial<Omit<Progress, 'id' | 'user_id' | 'topic'>>
      }
      learning_styles: {
        Row: LearningStyleData
        Insert: Omit<LearningStyleData, 'id' | 'detected_at' | 'updated_at'>
        Update: Partial<Omit<LearningStyleData, 'id' | 'user_id' | 'detected_at'>>
      }
      documents: {
        Row: Document
        Insert: Omit<Document, 'id' | 'uploaded_at' | 'processed_at'>
        Update: Partial<Omit<Document, 'id' | 'uploaded_by' | 'uploaded_at'>>
      }
    }
  }
}
