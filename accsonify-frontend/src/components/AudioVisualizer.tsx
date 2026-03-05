'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface AudioVisualizerProps {
    isRecording: boolean
}

export default function AudioVisualizer({ isRecording }: AudioVisualizerProps) {
    // A simple dummy visualizer using Framer Motion
    const numBars = 12

    return (
        <div className="flex items-center justify-center space-x-1 h-32 w-full my-6 bg-slate-900/50 rounded-2xl border border-white/5 overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-t from-indigo-500/10 to-transparent pointer-events-none" />

            {Array.from({ length: numBars }).map((_, i) => (
                <motion.div
                    key={i}
                    className="w-3 md:w-4 bg-indigo-500 rounded-full"
                    initial={{ height: 8 }}
                    animate={{
                        height: isRecording
                            ? [8, Math.random() * 60 + 20, Math.random() * 40 + 10, 8]
                            : 8,
                        backgroundColor: isRecording
                            ? ['#6366f1', '#a855f7', '#6366f1']
                            : '#475569'
                    }}
                    transition={{
                        duration: 0.5 + Math.random() * 0.5,
                        repeat: isRecording ? Infinity : 0,
                        repeatType: 'mirror',
                        ease: "easeInOut",
                        delay: isRecording ? Math.random() * 0.2 : 0,
                    }}
                />
            ))}
        </div>
    )
}
