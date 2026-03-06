'use client'

import React, { useState, useRef } from 'react'
import axios from 'axios'
import { Mic, Square, Play, Download, Languages, Sparkles, RefreshCw } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import AudioVisualizer from '@/components/AudioVisualizer'

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || '').replace(/\/$/, '')

const toApiUrl = (path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE}${normalizedPath}`
}

const toAbsoluteAudioUrl = (audioUrl: string) => {
  if (audioUrl.startsWith('http://') || audioUrl.startsWith('https://')) {
    return audioUrl
  }
  const normalizedPath = audioUrl.startsWith('/') ? audioUrl : `/${audioUrl}`
  return `${API_BASE}${normalizedPath}`
}

type DetectedAccent = {
  region: string
  confidence: number
}

export default function Home() {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const [detectedAccent, setDetectedAccent] = useState<DetectedAccent | null>(null)
  const [transcript, setTranscript] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const [targetAccent, setTargetAccent] = useState('indian')
  const [convertedAudioUrl, setConvertedAudioUrl] = useState<string | null>(null)
  const [isConverting, setIsConverting] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  // --- Recording Logic ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(audioBlob)
        setAudioUrl(url)
        await processInputAudio(audioBlob)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      setDetectedAccent(null)
      setTranscript(null)
      setConvertedAudioUrl(null)

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert("Could not access microphone. Ensure permissions are granted.")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
      setIsRecording(false)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  // --- API Calls ---
  const processInputAudio = async (blob: Blob) => {
    setIsProcessing(true)

    // We send two separate requests as defined by the architecture, though they could be optimized
    const formData = new FormData()
    formData.append('audio', blob, 'audio.webm')

    try {
      // 1. Detect Accent
      const detectRes = await axios.post(toApiUrl('/detect-accent'), formData)
      setDetectedAccent(detectRes.data)

      // 2. Transcribe
      const transcribeRes = await axios.post(toApiUrl('/transcribe'), formData)
      setTranscript(transcribeRes.data.text)
    } catch (error) {
      console.error('Error processing audio:', error)
      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail
        const detailMessage = typeof detail === 'string'
          ? detail
          : detail?.message || error.response?.data?.message
        alert(detailMessage || "Failed to process audio. Ensure backend is running.")
      } else {
        alert("Failed to process audio. Ensure backend is running.")
      }
    } finally {
      setIsProcessing(false)
    }
  }

  const handleConvert = async () => {
    if (!audioChunksRef.current.length) return

    setIsConverting(true)
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
    const formData = new FormData()
    formData.append('audio', audioBlob, 'audio.webm')
    formData.append('target_accent', targetAccent)

    try {
      const res = await axios.post(toApiUrl('/convert-accent'), formData)
      setConvertedAudioUrl(toAbsoluteAudioUrl(res.data.audio_url))
    } catch (error) {
      console.error('Conversion failed:', error)
      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail
        const detailMessage = typeof detail === 'string'
          ? detail
          : detail?.message || error.response?.data?.message
        alert(detailMessage || "Conversion failed.")
      } else {
        alert("Conversion failed.")
      }
    } finally {
      setIsConverting(false)
    }
  }

  const handleReset = () => {
    setAudioUrl(null)
    setDetectedAccent(null)
    setTranscript(null)
    setConvertedAudioUrl(null)
    setRecordingTime(0)
  }

  return (
    <main className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 relative z-10">

      {/* Header */}
      <div className="max-w-4xl mx-auto text-center mb-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-full mb-4"
        >
          <Languages className="w-8 h-8 text-indigo-400 mr-3" />
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            Accsonify
          </h1>
        </motion.div>
        <p className="mt-4 text-xl text-slate-300 max-w-2xl mx-auto">
          AI-powered Accent Detection and Seamless Voice Conversion Platform.
        </p>
      </div>

      <div className="max-w-3xl mx-auto space-y-8">

        {/* Step 1: Recorder */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-card p-8"
        >
          <div className="flex flex-col items-center">
            <h2 className="text-2xl font-bold mb-6 text-slate-100 flex items-center">
              <Mic className="mr-2 h-6 w-6 text-indigo-400" />
              Capture Speech
            </h2>

            <div className="text-4xl font-mono mb-8 text-indigo-300 font-light tracking-widest">
              {formatTime(recordingTime)}
            </div>

            <AudioVisualizer isRecording={isRecording} />

            <div className="mt-8 flex space-x-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="flex items-center px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-full font-semibold transition-all shadow-lg shadow-indigo-500/30 transform hover:scale-105"
                >
                  <Mic className="w-5 h-5 mr-2" /> Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex items-center px-8 py-4 bg-rose-600 hover:bg-rose-500 text-white rounded-full font-semibold transition-all shadow-lg shadow-rose-500/30 transform hover:scale-105 animate-pulse"
                >
                  <Square className="w-5 h-5 mr-2" /> Stop Recording
                </button>
              )}
            </div>

            {/* Show user audio player once recorded */}
            {audioUrl && !isRecording && (
              <div className="mt-8 w-full max-w-sm">
                <p className="text-sm text-slate-400 mb-2 text-center">Original Audio</p>
                <audio src={audioUrl} controls className="w-full h-10 rounded-full opacity-80" />
              </div>
            )}
          </div>
        </motion.div>

        {/* Processing State */}
        {isProcessing && (
          <div className="text-center py-6 text-indigo-300 animate-pulse flex justify-center items-center">
            <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
            Analyzing speech with AI models...
          </div>
        )}

        {/* Step 2: Results & Conversion (shown after processing) */}
        <AnimatePresence>
          {detectedAccent && transcript && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-8 border-l-4 border-l-teal-500"
            >
              <div className="grid md:grid-cols-2 gap-8">

                {/* Left: Detection Results */}
                <div>
                  <h3 className="text-xl font-bold mb-4 text-white">Detection Results</h3>

                  <div className="bg-slate-900/50 rounded-xl p-4 mb-4 border border-white/5">
                    <p className="text-sm text-slate-400">Classified Region</p>
                    <p className="text-2xl font-bold text-teal-400 mt-1">
                      {detectedAccent.region.replace('_', ' ')}
                    </p>
                    <div className="mt-2 w-full bg-slate-700 rounded-full h-2">
                      <div className="bg-teal-500 h-2 rounded-full" style={{ width: `${detectedAccent.confidence}%` }}></div>
                    </div>
                    <p className="text-xs text-right mt-1 text-slate-400">{detectedAccent.confidence}% Match</p>
                  </div>

                  <div className="bg-slate-900/50 rounded-xl p-4 border border-white/5">
                    <p className="text-sm text-slate-400 mb-2">Transcription</p>
                    <p className="text-slate-200 italic">"{transcript}"</p>
                  </div>
                </div>

                {/* Right: Conversion Settings */}
                <div className="flex flex-col justify-between">
                  <div>
                    <h3 className="text-xl font-bold mb-4 text-white">Target Accent</h3>
                    <div className="space-y-4">
                      <select
                        value={targetAccent}
                        onChange={(e) => setTargetAccent(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 text-white text-lg rounded-xl focus:ring-indigo-500 focus:border-indigo-500 block p-3 h-14"
                      >
                        <option value="indian">Indian (South Asia)</option>
                        <option value="british">British (Europe)</option>
                        <option value="american">American (North America)</option>
                        <option value="australian">Australian (Oceania)</option>
                      </select>
                    </div>
                  </div>

                  <button
                    onClick={handleConvert}
                    disabled={isConverting}
                    className={`mt-6 w-full flex items-center justify-center p-4 rounded-xl font-bold text-white transition-all shadow-lg ${isConverting
                        ? 'bg-indigo-600/50 cursor-not-allowed'
                        : 'bg-gradient-to-r from-teal-500 to-indigo-500 hover:from-teal-400 hover:to-indigo-400 hover:scale-[1.02]'
                      }`}
                  >
                    {isConverting ? (
                      <>
                        <RefreshCw className="animate-spin w-5 h-5 mr-3" /> Converting...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-3" /> Convert Magic
                      </>
                    )}
                  </button>
                </div>

              </div>
            </motion.div>
          )
          }
        </AnimatePresence >

        {/* Step 3: Converted Output Player */}
        <AnimatePresence>
          {
            convertedAudioUrl && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-card p-8 border-t-4 border-t-purple-500 flex flex-col items-center"
              >
                <h2 className="text-2xl font-bold mb-2 text-white flex items-center">
                  <Play className="mr-2 h-6 w-6 text-purple-400" />
                  Synthesized Speech
                </h2>
                <p className="text-slate-400 mb-6">Your voice properly matched and cloned with the target accent.</p>

                <div className="w-full max-w-lg bg-slate-900/60 p-4 rounded-2xl border border-white/5 mb-6">
                  <audio src={convertedAudioUrl} controls autoPlay className="w-full h-12" />
                </div>

                <div className="flex space-x-4">
                  <a
                    href={convertedAudioUrl}
                    download="accsonify-converted.mp3"
                    className="flex items-center px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg text-white font-medium transition-colors"
                  >
                    <Download className="w-4 h-4 mr-2" /> Download
                  </a>
                  <button
                    onClick={handleReset}
                    className="flex items-center px-6 py-3 border border-slate-600 hover:border-slate-400 rounded-lg text-white font-medium transition-colors"
                  >
                    Record New
                  </button>
                </div>
              </motion.div>
            )
          }
        </AnimatePresence >

      </div >
    </main >
  )
}
