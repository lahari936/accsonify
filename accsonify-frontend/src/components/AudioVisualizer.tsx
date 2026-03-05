'use client'

import React, { useMemo } from 'react'
import { motion } from 'framer-motion'

interface AudioVisualizerProps {
    isRecording: boolean
}

export default function AudioVisualizer({ isRecording }: AudioVisualizerProps) {
    // A simple dummy visualizer using Framer Motion
    const numBars = 12
    const barConfigs = useMemo(
        () =>
            Array.from({ length: numBars }, (_, i) => {
                const peakHeight = 24 + ((i * 17) % 56)
                const midHeight = 12 + ((i * 11) % 34)
                const duration = 0.55 + ((i % 5) * 0.1)
                const delay = (i % 6) * 0.03

                return {
                    peakHeight,
                    midHeight,
                    duration,
                    delay,
                }
            }),
        [numBars]
    )

    return (
        <div className="flex items-center justify-center space-x-1 h-32 w-full my-6 bg-slate-900/50 rounded-2xl border border-white/5 overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-t from-indigo-500/10 to-transparent pointer-events-none" />

            {barConfigs.map((bar, i) => (
                <motion.div
                    key={i}
                    className="w-3 md:w-4 bg-indigo-500 rounded-full"
                    initial={{ height: 8 }}
                    animate={{
                        height: isRecording
                            ? [8, bar.peakHeight, bar.midHeight, 8]
                            : 8,
                        backgroundColor: isRecording
                            ? ['#6366f1', '#a855f7', '#6366f1']
                            : '#475569'
                    }}
                    transition={{
                        duration: bar.duration,
                        repeat: isRecording ? Infinity : 0,
                        repeatType: 'mirror',
                        ease: "easeInOut",
                        delay: isRecording ? bar.delay : 0,
                    }}
                />
            ))}
        </div>
    )
}
