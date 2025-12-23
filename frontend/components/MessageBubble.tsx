'use client'

import { useState } from 'react'
import { ChatMessage } from '@/types'
import { CheckCircle2, AlertCircle, XCircle, Target, BarChart3, Lightbulb, BookOpen } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import RAGSourcesModal from './RAGSourcesModal'

interface MessageBubbleProps {
  message: ChatMessage
  isStreaming?: boolean
}

export default function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const [showRAGSources, setShowRAGSources] = useState(false)
  const hasRAGSources = !isUser && message.metadata?.rag_sources && message.metadata.rag_sources.length > 0

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-3xl rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-white text-gray-900 border border-gray-200'
        }`}
      >
        <div className="break-words markdown-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[]}
            // Disable code block detection for content that looks like curriculum
            skipHtml={false}
            components={{
              // Customize pre blocks (wraps code blocks) - must be before code component
              pre: ({ children, ...props }: any) => {
                // Check if this pre contains a code block with a language class
                const hasCodeBlock = children && 
                  typeof children === 'object' && 
                  children.props && 
                  children.props.className &&
                  /language-(\w+)/.test(children.props.className)
                
                if (hasCodeBlock) {
                  return (
                    <pre className={`rounded-md p-3 overflow-x-auto border my-2 ${
                      isUser 
                        ? 'bg-primary-700 border-primary-500' 
                        : 'bg-gray-100 border-gray-300'
                    }`} {...props}>
                      {children}
                    </pre>
                  )
                }
                // For other pre blocks (if any)
                return <pre className="my-2" {...props}>{children}</pre>
              },
              // Customize code blocks
              code: ({ className, children, ...props }: any) => {
                const match = /language-(\w+)/.exec(className || '')
                const isInline = !match
                return isInline ? (
                  <code 
                    className={`px-1.5 py-0.5 rounded text-sm font-mono ${
                      isUser 
                        ? 'bg-primary-700 text-primary-100' 
                        : 'bg-gray-100 text-primary-600'
                    }`} 
                    {...props}
                  >
                    {children}
                  </code>
                ) : (
                  // For block code, ReactMarkdown already wraps in <pre> (handled by pre component above)
                  // Just return the code element with proper styling
                  <code 
                    className={`${className || ''} block whitespace-pre-wrap font-mono text-sm`} 
                    {...props}
                  >
                    {children}
                  </code>
                )
              },
              // Customize links
              a: ({ ...props }: any) => (
                <a 
                  className={isUser ? 'text-primary-100 hover:text-white underline' : 'text-primary-600 hover:text-primary-700 underline'} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  {...props} 
                />
              ),
              // Customize lists with better spacing
              ul: ({ ...props }: any) => (
                <ul className={`list-disc pl-6 space-y-2 my-3 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              ol: ({ ...props }: any) => (
                <ol className={`list-decimal pl-6 space-y-2 my-3 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize list items
              li: ({ ...props }: any) => (
                <li className={`${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize blockquotes
              blockquote: ({ ...props }: any) => (
                <blockquote 
                  className={`border-l-4 pl-4 italic my-3 ${
                    isUser 
                      ? 'border-primary-300 text-primary-100' 
                      : 'border-gray-300 text-gray-700'
                  }`} 
                  {...props} 
                />
              ),
              // Customize paragraphs with better spacing
              p: ({ ...props }: any) => (
                <p className={`my-3 leading-relaxed ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize headings with better hierarchy and spacing
              h1: ({ ...props }: any) => (
                <h1 className={`text-3xl font-bold my-4 pb-2 border-b ${isUser ? 'text-white border-primary-400' : 'text-gray-900 border-gray-300'}`} {...props} />
              ),
              h2: ({ ...props }: any) => (
                <h2 className={`text-2xl font-bold my-4 mt-6 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              h3: ({ ...props }: any) => (
                <h3 className={`text-xl font-semibold my-3 mt-5 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              h4: ({ ...props }: any) => (
                <h4 className={`text-lg font-semibold my-2 mt-4 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize strong/bold text
              strong: ({ ...props }: any) => (
                <strong className={`font-bold ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize emphasis/italic text
              em: ({ ...props }: any) => (
                <em className={`italic ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              // Customize horizontal rules
              hr: ({ ...props }: any) => (
                <hr className={`my-6 border-t ${isUser ? 'border-primary-400' : 'border-gray-300'}`} {...props} />
              ),
              // Customize tables
              table: ({ ...props }: any) => (
                <div className="overflow-x-auto my-4">
                  <table className={`min-w-full border-collapse border ${isUser ? 'border-primary-400' : 'border-gray-300'}`} {...props} />
                </div>
              ),
              thead: ({ ...props }: any) => (
                <thead className={isUser ? 'bg-primary-700' : 'bg-gray-100'} {...props} />
              ),
              tbody: ({ ...props }: any) => (
                <tbody {...props} />
              ),
              tr: ({ ...props }: any) => (
                <tr className={`border-b ${isUser ? 'border-primary-400' : 'border-gray-300'}`} {...props} />
              ),
              th: ({ ...props }: any) => (
                <th className={`px-4 py-2 text-left font-semibold ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
              td: ({ ...props }: any) => (
                <td className={`px-4 py-2 ${isUser ? 'text-white' : 'text-gray-900'}`} {...props} />
              ),
            }}
          >
            {(() => {
              // Strip markdown code fences if the entire content is wrapped in them
              let content = String(message.content || '').trim()
              
              // More robust detection: handle various code fence formats
              // Pattern 1: ```markdown\n...\n```
              const markdownFencePattern = /^```markdown\s*\n([\s\S]*?)\n```\s*$/m
              // Pattern 2: ```\n...\n``` (generic)
              const genericFencePattern = /^```\w*\s*\n([\s\S]*?)\n```\s*$/m
              // Pattern 3: ```markdown...``` (no newlines)
              const markdownFenceNoNewline = /^```markdown\s*([\s\S]*?)```\s*$/m
              // Pattern 4: ```...``` (generic, no newlines)
              const genericFenceNoNewline = /^```\w*\s*([\s\S]*?)```\s*$/m
              
              // Try to extract content from code fences
              let match = markdownFencePattern.exec(content)
              if (match) {
                content = match[1].trim()
              } else {
                match = genericFencePattern.exec(content)
                if (match) {
                  content = match[1].trim()
                } else {
                  match = markdownFenceNoNewline.exec(content)
                  if (match) {
                    content = match[1].trim()
                  } else {
                    match = genericFenceNoNewline.exec(content)
                    if (match) {
                      content = match[1].trim()
                    }
                  }
                }
              }
              
              // Additional check: if content starts with markdown header (#) and looks like curriculum,
              // ensure it's not being treated as code. This handles edge cases where code fences
              // might be detected incorrectly or partially.
              if (content.startsWith('#') && content.includes('Learning Plan')) {
                // This is definitely markdown content, not code
                // Ensure no code fences remain
                content = content.replace(/^```+\w*\s*/gm, '').replace(/```+\s*$/gm, '')
              }
              
              return content
            })()}
          </ReactMarkdown>
          {isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
          )}
        </div>

        {/* Metadata */}
        {message.metadata && !isUser && (
          <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
            {/* Learning Style */}
            {message.metadata.learning_style_detected && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <span className="text-lg">🎨</span>
                <span>
                  Learning style detected: <strong>{message.metadata.learning_style_detected}</strong>
                </span>
              </div>
            )}

            {/* Adaptation Event */}
            {message.metadata.adaptation_event && (
              <div className="flex items-center space-x-2 text-sm">
                {message.metadata.adaptation_event === 'excellent_performance' && (
                  <>
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span className="text-green-700">⭐ Excellent performance!</span>
                  </>
                )}
                {message.metadata.adaptation_event === 'struggling' && (
                  <>
                    <Lightbulb className="h-4 w-4 text-yellow-500" />
                    <span className="text-yellow-700">💡 Providing additional support</span>
                  </>
                )}
                {message.metadata.adaptation_event === 'prerequisites_needed' && (
                  <>
                    <AlertCircle className="h-4 w-4 text-blue-500" />
                    <span className="text-blue-700">📚 Prerequisites identified</span>
                  </>
                )}
                {message.metadata.adaptation_event === 'moderate_performance' && (
                  <>
                    <BarChart3 className="h-4 w-4 text-gray-500" />
                    <span className="text-gray-700">📈 Continuing at current pace</span>
                  </>
                )}
              </div>
            )}

            {/* Evaluation */}
            {message.metadata.evaluation && (
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  {message.metadata.evaluation.correctness === 'correct' && (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                  {message.metadata.evaluation.correctness === 'partially_correct' && (
                    <AlertCircle className="h-4 w-4 text-yellow-500" />
                  )}
                  {message.metadata.evaluation.correctness === 'incorrect' && (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                  <span className="text-gray-600">
                    {message.metadata.evaluation.correctness.replace('_', ' ')}
                  </span>
                </div>
                <div className="flex items-center space-x-1">
                  <Target className="h-4 w-4 text-gray-500" />
                  <span className="text-gray-600">
                    {message.metadata.evaluation.depth} depth
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* RAG Sources Button */}
        {hasRAGSources && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <button
              onClick={() => setShowRAGSources(true)}
              className="flex items-center space-x-2 text-sm text-indigo-600 hover:text-indigo-700 transition-colors"
            >
              <BookOpen className="h-4 w-4" />
              <span>View RAG Sources ({message.metadata?.rag_sources?.length ?? 0})</span>
            </button>
          </div>
        )}

        {/* Timestamp */}
        <div className={`text-xs mt-2 ${isUser ? 'text-primary-100' : 'text-gray-400'}`}>
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>

      {/* RAG Sources Modal */}
      {hasRAGSources && (
        <RAGSourcesModal
          chunks={message.metadata?.rag_sources ?? []}
          isOpen={showRAGSources}
          onClose={() => setShowRAGSources(false)}
        />
      )}
    </div>
  )
}

