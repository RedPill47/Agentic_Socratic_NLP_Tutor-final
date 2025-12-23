'use client'

import { useState, useEffect } from 'react'
import { X, FileText, Loader2, BookOpen } from 'lucide-react'
import { AuthenticatedAPI } from '@/lib/api-authenticated'

interface RAGChunk {
  content: string
  source_file: string
  page_number?: number
  slide_title?: string
  topic?: string
  difficulty?: string
}

interface RAGMaterialsModalProps {
  isOpen: boolean
  onClose: () => void
  sessionId: string | null
  sourceFiles: string[]
  currentTopic: string | null
}

export default function RAGMaterialsModal({
  isOpen,
  onClose,
  sessionId,
  sourceFiles,
  currentTopic,
}: RAGMaterialsModalProps) {
  const [chunks, setChunks] = useState<RAGChunk[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen && sessionId && sourceFiles.length > 0) {
      loadRAGMaterials()
    } else {
      setChunks([])
      setError(null)
    }
  }, [isOpen, sessionId, sourceFiles])

  const loadRAGMaterials = async () => {
    if (!sessionId) return

    setLoading(true)
    setError(null)

    try {
      const materials = await AuthenticatedAPI.getRAGMaterials(sessionId)
      setChunks(materials.chunks || [])
    } catch (err: any) {
      setError(err.message || 'Failed to load RAG materials')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  // Group chunks by source file
  const chunksByFile = chunks.reduce((acc, chunk) => {
    const file = chunk.source_file || 'Unknown Source'
    if (!acc[file]) {
      acc[file] = []
    }
    acc[file].push(chunk)
    return acc
  }, {} as Record<string, RAGChunk[]>)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <BookOpen className="h-6 w-6 text-indigo-600" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900">RAG Materials</h2>
              <p className="text-sm text-gray-500">
                {currentTopic ? `Topic: ${currentTopic}` : 'Source materials used in this session'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Close"
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
              <span className="ml-3 text-gray-600">Loading RAG materials...</span>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-red-600 mb-2">{error}</p>
              <button
                onClick={loadRAGMaterials}
                className="text-indigo-600 hover:text-indigo-700 text-sm"
              >
                Try again
              </button>
            </div>
          ) : chunks.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No RAG materials found for this session.</p>
              <p className="text-sm text-gray-400 mt-2">
                Materials will appear after you send messages that trigger RAG retrieval.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Summary */}
              <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Source Files:</span>
                    <span className="ml-2 font-semibold text-gray-900">{sourceFiles.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Total Chunks:</span>
                    <span className="ml-2 font-semibold text-gray-900">{chunks.length}</span>
                  </div>
                </div>
              </div>

              {/* Chunks grouped by file */}
              {Object.entries(chunksByFile).map(([fileName, fileChunks]) => (
                <div key={fileName} className="border border-gray-200 rounded-lg overflow-hidden">
                  {/* File Header */}
                  <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                    <div className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-indigo-600" />
                      <h3 className="font-semibold text-gray-900">{fileName}</h3>
                      <span className="ml-auto text-xs text-gray-500">
                        {fileChunks.length} chunk{fileChunks.length !== 1 ? 's' : ''}
                      </span>
                    </div>
                  </div>

                  {/* Chunks */}
                  <div className="divide-y divide-gray-200">
                    {fileChunks.map((chunk, index) => (
                      <div key={index} className="p-4 hover:bg-gray-50 transition-colors">
                        {/* Chunk Metadata */}
                        <div className="flex items-center gap-3 mb-2 text-xs text-gray-500">
                          {chunk.page_number && (
                            <span>Page {chunk.page_number}</span>
                          )}
                          {chunk.slide_title && (
                            <span className="text-gray-400">•</span>
                          )}
                          {chunk.slide_title && (
                            <span className="italic">"{chunk.slide_title}"</span>
                          )}
                          {chunk.topic && (
                            <>
                              <span className="text-gray-400">•</span>
                              <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                                {chunk.topic}
                              </span>
                            </>
                          )}
                          {chunk.difficulty && (
                            <>
                              <span className="text-gray-400">•</span>
                              <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded capitalize">
                                {chunk.difficulty}
                              </span>
                            </>
                          )}
                        </div>

                        {/* Chunk Content */}
                        <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                          {chunk.content}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

