'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/lib/hooks/useAuth'
import Link from 'next/link'

export default function Home() {
  const { user, profile, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && user) {
      if (profile?.role === 'teacher') {
        router.push('/teacher')
      } else {
        router.push('/chat')
      }
    }
  }, [user, loading, router, profile])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-6xl font-bold text-gray-900 mb-6">
            Master NLP with
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">
              {' '}AI Guidance
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-12 max-w-2xl mx-auto">
            Learn Natural Language Processing through personalized, Socratic dialogue.
            Your AI tutor adapts to your learning style and guides you step by step.
          </p>

          {/* CTA Buttons */}
          <div className="flex gap-4 justify-center">
            <Link
              href="/signup"
              className="px-8 py-4 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 transition-colors shadow-lg hover:shadow-xl"
            >
              Get Started
            </Link>
            <Link
              href="/login"
              className="px-8 py-4 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-50 transition-colors shadow-lg hover:shadow-xl border-2 border-indigo-600"
            >
              Sign In
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mt-24 max-w-5xl mx-auto">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="text-4xl mb-4">🎯</div>
            <h3 className="text-xl font-semibold mb-2">Personalized Learning</h3>
            <p className="text-gray-600">
              Adapts to your learning style and pace with real-time adjustments
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="text-4xl mb-4">💬</div>
            <h3 className="text-xl font-semibold mb-2">Socratic Method</h3>
            <p className="text-gray-600">
              Learn through guided questions, not direct answers
            </p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-md">
            <div className="text-4xl mb-4">📊</div>
            <h3 className="text-xl font-semibold mb-2">Track Progress</h3>
            <p className="text-gray-600">
              Monitor your learning journey with detailed analytics
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

