'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface PerformanceChartProps {
  data: Array<{
    interaction: number
    correctness: number
    depth: number
  }>
}

export default function PerformanceChart({ data }: PerformanceChartProps) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="interaction" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Line
          type="monotone"
          dataKey="correctness"
          stroke="#0ea5e9"
          strokeWidth={2}
          name="Correctness"
        />
        <Line
          type="monotone"
          dataKey="depth"
          stroke="#10b981"
          strokeWidth={2}
          name="Depth"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

