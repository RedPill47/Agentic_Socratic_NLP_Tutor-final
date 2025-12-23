import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '@/lib/hooks/useAuth'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Agentic Socratic NLP Tutor',
  description: 'AI-powered tutoring system for Natural Language Processing',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  )
}

