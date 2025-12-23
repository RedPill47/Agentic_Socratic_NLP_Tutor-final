'use client'

import { X, FileText, BookOpen } from 'lucide-react'
import { RAGChunk } from '@/types'

interface RAGSourcesModalProps {
  chunks: RAGChunk[]
  isOpen: boolean
  onClose: () => void
}

export default function RAGSourcesModal({ chunks, isOpen, onClose }: RAGSourcesModalProps) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onClick={onClose}>
      <div 
        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <BookOpen className="h-6 w-6 text-indigo-600" />
            <h2 className="text-2xl font-bold text-gray-900">RAG Sources Used</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {chunks.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No RAG sources available for this message.</p>
          ) : (
            <div className="space-y-6">
              {chunks.map((chunk, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow"
                >
                  {/* Source Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <FileText className="h-5 w-5 text-indigo-600" />
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          {chunk.source_file || 'Unknown Source'}
                        </h3>
                        <div className="flex items-center space-x-4 text-sm text-gray-500 mt-1">
                          {chunk.page_number && (
                            <span>Page {chunk.page_number}</span>
                          )}
                          {chunk.topic && (
                            <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded">
                              {chunk.topic}
                            </span>
                          )}
                          {chunk.difficulty && (
                            <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded capitalize">
                              {chunk.difficulty}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Slide Title (if available) */}
                  {chunk.slide_title && (
                    <div className="mb-3">
                      <p className="text-sm font-medium text-gray-700 italic">
                        "{chunk.slide_title}"
                      </p>
                    </div>
                  )}

                  {/* Content */}
                  <div className="prose prose-sm max-w-none">
                    <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                      {chunk.content}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <p className="text-sm text-gray-600 text-center">
            These are the RAG chunks the agent used to generate this response
          </p>
        </div>
      </div>
    </div>
  )
}

