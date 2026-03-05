import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Accsonify | AI Accent Conversion',
  description: 'Real-time accent detection and conversion platform.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.variable} antialiased bg-slate-900 text-slate-50 selection:bg-indigo-500/30`}>
        {children}
      </body>
    </html>
  )
}
