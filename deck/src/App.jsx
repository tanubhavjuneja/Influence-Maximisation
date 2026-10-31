import React, { useEffect, useState } from 'react'
import { SLIDES } from './slides.jsx'

const fromHash = () => {
  const h = Number(window.location.hash.replace('#', ''))
  return Number.isInteger(h) && h >= 0 && h < SLIDES.length ? h : 0
}

export default function App() {
  const total = SLIDES.length
  const [i, setI] = useState(fromHash)

  useEffect(() => {
    const onHash = () => setI(fromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') setI((v) => Math.min(total - 1, v + 1))
      if (e.key === 'ArrowLeft') setI((v) => Math.max(0, v - 1))
      if (e.key === 'Home') setI(0)
      if (e.key === 'End') setI(total - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [total])

  const go = (j) => {
    setI(j)
    window.location.hash = String(j)
  }

  const Slide = SLIDES[i]
  return (
    <div className="stage">
      <Slide />
      <div className="nav">
        <button onClick={() => go(Math.max(0, i - 1))}>← prev</button>
        <div className="dots">
          {SLIDES.map((_, j) => (
            <div key={j} className={`dot ${j === i ? 'on' : ''}`} onClick={() => go(j)} />
          ))}
        </div>
        <span className="count">{i + 1} / {total}</span>
        <button onClick={() => go(Math.min(total - 1, i + 1))}>next →</button>
      </div>
    </div>
  )
}