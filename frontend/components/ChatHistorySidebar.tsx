'use client'

import { useMemo, useState, type KeyboardEvent } from 'react'
import { Plus, Clock, ChevronLeft, ChevronRight, MessageSquare, Trash } from 'lucide-react'

export interface ChatHistoryItem {
  sessionId: string
  title: string
  lastActivity?: string
  isMock?: boolean
}

interface ChatHistorySidebarProps {
  items: ChatHistoryItem[]
  activeSessionId: string | null
  onSelectSession?: (sessionId: string) => void
  onNewChat: () => void
  disableSelection?: boolean
  onDeleteSession?: (sessionId: string) => void
}

export default function ChatHistorySidebar({
  items,
  activeSessionId,
  onSelectSession,
  onNewChat,
  disableSelection = false,
  onDeleteSession,
}: ChatHistorySidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const widthClass = isCollapsed ? 'w-16' : 'w-72'

  const formattedItems = useMemo(
    () =>
      items.map((item) => ({
        ...item,
        subtitle: item.lastActivity
          ? new Date(item.lastActivity).toLocaleDateString(undefined, {
              month: 'short',
              day: 'numeric',
            })
          : 'Draft',
      })),
    [items]
  )

  return (
    <div
      className={`relative flex h-screen flex-col border-r border-gray-200 bg-slate-50/90 backdrop-blur transition-all duration-300 ease-in-out ${widthClass}`}
    >
      {/* Top controls */}
      <div className="sticky top-0 z-10 border-b border-gray-200 bg-slate-50/90 px-3 py-3">
        <div className="flex items-center justify-between gap-2">
          <button
            onClick={onNewChat}
            className="inline-flex h-11 flex-1 items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            title="Start a new chat session"
          >
            <Plus className="h-4 w-4" />
            {!isCollapsed && <span>New Chat</span>}
          </button>
          <button
            onClick={() => setIsCollapsed((prev) => !prev)}
            className="flex h-11 w-11 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-600 shadow-sm transition hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            aria-label={isCollapsed ? 'Expand chat history' : 'Collapse chat history'}
          >
            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* History list */}
      <div className="flex-1 space-y-1 overflow-y-auto px-2 py-4">
        {formattedItems.map((item) => {
          const isActive = activeSessionId === item.sessionId
          const handleClick = () => {
            if (!disableSelection && onSelectSession) {
              onSelectSession(item.sessionId)
            }
          }

          const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              handleClick()
            }
          }

          return (
            <div
              key={item.sessionId}
              role="button"
              tabIndex={disableSelection ? -1 : 0}
              onClick={handleClick}
              onKeyDown={handleKeyDown}
              className={`group relative flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-gray-800 hover:bg-indigo-50'
              } ${disableSelection ? 'cursor-not-allowed opacity-70' : 'cursor-pointer'}`}
            >
              <div
                className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg border text-indigo-600 ${
                  isActive ? 'border-indigo-400 bg-white/20 text-white' : 'border-gray-200 bg-white'
                }`}
              >
                <MessageSquare className="h-5 w-5" />
              </div>
              {!isCollapsed && (
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <p className="truncate text-sm font-semibold">{item.title}</p>
                    {item.isMock && (
                      <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-indigo-700">
                        Mock
                      </span>
                    )}
                  </div>
                  <div
                    className={`mt-1 flex items-center gap-1 text-xs ${
                      isActive ? 'text-indigo-100' : 'text-gray-500'
                    }`}
                  >
                    <Clock className="h-3 w-3" />
                    <span>{item.subtitle}</span>
                  </div>
                </div>
              )}
              {!isCollapsed && onDeleteSession && (
                <button
                  className={`ml-1 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md border text-gray-500 transition hover:text-red-600 hover:border-red-200 ${
                    isActive ? 'border-white/40 bg-white/10 text-white hover:bg-red-500/10' : 'border-gray-200 bg-white'
                  }`}
                  aria-label="Delete chat"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(item.sessionId)
                  }}
                >
                  <Trash className="h-4 w-4" />
                </button>
              )}
              {isCollapsed && isActive && <span className="sr-only">Active chat</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}
