'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, BookOpen, HelpCircle } from 'lucide-react'
import { ChatMessage, StateSummary } from '@/types'
import { AuthenticatedAPI } from '@/lib/api-authenticated'
import MessageBubble from './MessageBubble'

interface ChatInterfaceProps {
  sessionId: string
  userId?: string
  onStateUpdate: (state: StateSummary) => void
  currentState?: StateSummary | null
}

export default function ChatInterface({ sessionId, userId, onStateUpdate, currentState }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const isNearBottom = () => {
    if (!chatContainerRef.current) return true
    const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current
    // Consider "near bottom" if within 100px of bottom
    return scrollHeight - scrollTop - clientHeight < 100
  }

  const scrollToBottom = (instant = false) => {
    if (messagesEndRef.current && isNearBottom()) {
      if (instant) {
        messagesEndRef.current.scrollIntoView({ behavior: 'auto' })
      } else {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }

  // Auto-scroll when messages change (only if user is near bottom)
  useEffect(() => {
    if (messages.length > 0 && isNearBottom()) {
      // Use setTimeout to ensure DOM is updated
      setTimeout(() => scrollToBottom(false), 100)
    }
  }, [messages.length])

  // Auto-scroll during streaming (only if user is near bottom)
  useEffect(() => {
    if (streamingContent && isNearBottom()) {
      // Instant scroll during streaming for real-time feel
      scrollToBottom(true)
    }
  }, [streamingContent])

  // Load conversation history when component mounts or sessionId changes
  useEffect(() => {
    const loadHistory = async () => {
      if (!sessionId) return
      
      try {
        setLoadingHistory(true)
        // Reset messages when loading new session
        setMessages([])
        
        const historyMessages = await AuthenticatedAPI.getSessionMessages(sessionId)

        // Convert to ChatMessage format
        const chatMessages: ChatMessage[] = historyMessages.map((msg) => ({
          role: msg.role as 'user' | 'assistant' | 'system',
          content: msg.content,
          timestamp: new Date(msg.created_at),
          metadata: msg.metadata,
        }))

        // If we have messages, use them
        if (chatMessages.length > 0) {
          setMessages(chatMessages)
        } else {
          // If no messages exist, get welcome message
          try {
            const welcomeResponse = await AuthenticatedAPI.getWelcomeMessage(sessionId)
            const welcomeMessage: ChatMessage = {
              role: 'assistant',
              content: welcomeResponse.content,
              timestamp: new Date(),
            }
            setMessages([welcomeMessage])
          } catch (error) {
            // Continue without welcome message if it fails
            setMessages([])
          }
        }
      } catch (error) {
        // Start with empty messages if history fails to load
        
        // Try to get welcome message even if history loading failed
        try {
          const welcomeResponse = await AuthenticatedAPI.getWelcomeMessage(sessionId)
          const welcomeMessage: ChatMessage = {
            role: 'assistant',
            content: welcomeResponse.content,
            timestamp: new Date(),
          }
          setMessages([welcomeMessage])
        } catch (welcomeError) {
          setMessages([])
        }
      } finally {
        setLoadingHistory(false)
      }
    }

    if (sessionId) {
      loadHistory()
    }
  }, [sessionId])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setStreamingContent('')
    
    // Scroll to bottom when user sends a message
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
      }
    }, 50)

    try {
      let fullResponse = ''
      let metadata: any = null

      await AuthenticatedAPI.chatStream(
        {
          content: input,
          session_id: sessionId,
        },
        (token) => {
          // Accumulate tokens as they arrive
          fullResponse += token
          setStreamingContent(fullResponse)
        },
        (meta) => {
          // Store metadata when stream completes
          metadata = meta
        },
        (error: any) => {
          setIsLoading(false)
          setStreamingContent('')
          // Show error message
          const errorMessage: ChatMessage = {
            role: 'assistant',
            content: `Error: ${error?.message || 'Unknown error'}. Please try again.`,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, errorMessage])
        },
        (state, message) => {
          // Show state updates (e.g., "Fetching materials...")
          if (message && !fullResponse) {
            // Only show state message if we haven't started receiving content yet
            setStreamingContent(message)
          }
        }
      )

      // Stream completed - stop loading and clear streaming content
      setIsLoading(false)
      setStreamingContent('')

      // Add assistant message (only if we got a response)
      if (fullResponse.trim()) {
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: fullResponse,
          timestamp: new Date(),
          metadata: metadata
            ? {
                learning_style_detected: metadata.learning_style?.primary_style,
                adaptation_event: metadata.adaptation_event,
                state: metadata.state,
                topic: metadata.topic,
              } as any
            : undefined,
        }

        setMessages((prev) => [...prev, assistantMessage])
        
        // Fetch updated state after message to get source files and topic
        try {
          const updatedState = await AuthenticatedAPI.getState(sessionId)
          onStateUpdate(updatedState)
        } catch (error) {
          // Silently fail - state update is not critical
        }
      }
    } catch (error: any) {
      setIsLoading(false)

      // Show error message to user
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${error.message}. Please make sure you're logged in and try again.`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleCreateCurriculum = () => {
    if (isLoading) return
    
    const topic = currentState?.current_topic
    const curriculumRequest = topic 
      ? `Create a learning plan for ${topic}`
      : 'Create a learning plan for '
    
    // Set input field with template text
    setInput(curriculumRequest)
    
    // Focus the input field so user can edit it
    setTimeout(() => {
      inputRef.current?.focus()
      // Select the text after "for " so user can easily replace it
      if (inputRef.current) {
        const selectStart = curriculumRequest.length
        inputRef.current.setSelectionRange(selectStart, selectStart)
      }
    }, 0)
  }

  if (loadingHistory) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-600 mx-auto mb-2" />
          <p className="text-sm text-gray-500">Loading conversation...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col bg-white overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">NLP Learning Chat</h2>
            <p className="text-sm text-gray-500">Ask questions and learn through Socratic dialogue</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative group">
              <button
                onClick={handleCreateCurriculum}
                disabled={isLoading}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
                title="Generate a personalized learning curriculum"
              >
                <BookOpen className="h-4 w-4" />
                <span className="hidden sm:inline">Create Curriculum</span>
                <span className="sm:hidden">Curriculum</span>
              </button>
              {/* Tooltip */}
              <div className="absolute right-0 top-full mt-2 w-64 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                <div className="flex items-start gap-2">
                  <HelpCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-semibold mb-1">Personalized Learning Plan</p>
                    <p className="text-gray-300">
                      Get a customized curriculum based on your current knowledge, learning style, and goals. 
                      Takes 15-45 seconds to generate.
                    </p>
                  </div>
                </div>
                <div className="absolute -top-1 right-4 w-2 h-2 bg-gray-900 rotate-45"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto px-6 py-4 space-y-4 scroll-smooth"
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Welcome to Your NLP Journey!
              </h3>
              <p className="text-gray-600 mb-4">
                Start by introducing yourself or asking about an NLP topic you'd like to
                learn.
              </p>
              <div className="text-left space-y-2 mb-6">
                <p className="text-sm text-gray-500">
                  <strong>Example:</strong> "Hi, I'm new to NLP and want to learn about
                  tokenization"
                </p>
                <p className="text-sm text-gray-500">
                  <strong>Example:</strong> "What is Named Entity Recognition?"
                </p>
              </div>
              <div className="border-t border-gray-200 pt-6">
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Or get a personalized learning plan:
                </p>
                <button
                  onClick={handleCreateCurriculum}
                  disabled={isLoading}
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
                >
                  <BookOpen className="h-5 w-5" />
                  Create Learning Curriculum
                </button>
                <p className="text-xs text-gray-500 mt-2">
                  Get a customized study plan based on your goals and current knowledge
                </p>
              </div>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}

        {/* Streaming message */}
        {streamingContent && (
          <MessageBubble
            message={{
              role: 'assistant',
              content: streamingContent,
              timestamp: new Date(),
            }}
            isStreaming={isLoading}
          />
        )}

        {/* Loading indicator */}
        {isLoading && !streamingContent && (
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center">
                <span className="text-sm">🤖</span>
              </div>
            </div>
            <div className="flex-1 bg-gray-50 rounded-lg p-4">
              <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white px-6 py-4 flex-shrink-0">
        <div className="flex space-x-4">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1 rounded-lg border border-gray-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 text-white rounded-lg px-6 py-3 font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <>
                <Send className="h-5 w-5" />
                <span>Send</span>
              </>
            )}
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
