import React from 'react'

export const KPI = ({ v, l, tone }) => (
  <div className={`kpi ${tone || ''}`}>
    <div className="v">{v}</div>
    <div className="l">{l}</div>
  </div>
)

export const Table = ({ cols, rows, caption, rowClass }) => (
  <table className="tbl">
    {caption ? <caption>{caption}</caption> : null}
    <thead>
      <tr>{cols.map((c) => <th key={c} className={c.match(/^[+\-−0-9]/) ? 'num' : ''}>{c}</th>)}</tr>
    </thead>
    <tbody>
      {rows.map((r, i) => (
        <tr key={i} className={rowClass ? rowClass(r) : ''}>
          {cols.map((c) => {
            const cell = r[c]
            const cls = typeof cell === 'object' && cell.cls ? cell.cls : ''
            return <td key={c} className={cls}>{typeof cell === 'object' ? cell.v : cell}</td>
          })}
        </tr>
      ))}
    </tbody>
  </table>
)

export const Head = ({ kicker, title, subtitle, tag }) => (
  <div className="head">
    <div>
      <div className="kicker">{kicker}</div>
      <div className="title">{title}</div>
      {subtitle ? <div className="subtitle">{subtitle}</div> : null}
    </div>
    {tag ? <div className="tag">{tag}</div> : null}
  </div>
)

export const Foot = ({ src, n, i }) => (
  <div className="foot">
    <span className="src">{src}</span>
    <span>{i + 1} / {n}</span>
  </div>
)

export const Legend = ({ items }) => (
  <div className="legend">
    {items.map((it) => (
      <div className="li" key={it.label}>
        <div className="sw" style={{ background: it.color }} />
        <span>{it.label}</span>
      </div>
    ))}
  </div>
)

/**
 * Forest / dumbell plot in pure SVG: one row per graph, markers for
 * two paired differences + 95% bootstrap CI whiskers.
 */
export const Forest = ({ rows, xlab, ci, width = 520, height, min, max, loKey = 'abc', hiKey = 'sa' }) => {
  const n = rows.length
  const H = height || Math.max(220, n * 26 + 46)
  const padL = 74, padR = 22, padT = 10, padB = 34
  const lo = min != null ? min : Math.min(0, ...rows.map((r) => r[loKey]), ...rows.map((r) => (ci ? ci[0] : 0)))
  const hi = max != null ? max : Math.max(0, ...rows.map((r) => r[loKey]), ...rows.map((r) => (ci ? ci[1] : 0)))
  const X = (v) => padL + ((v - lo) / (hi - lo)) * (width - padL - padR)
  const band = (hi - lo) / (width - padL - padR)
  const rowH = (H - padT - padB) / n
  return (
    <svg width={width} height={H} style={{ fontFamily: 'var(--sans)' }}>
      {[0, lo / 2, hi / 2].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={padT - 4} y2={H - padB} stroke="#e4e2da" />
          <text x={X(v)} y={H - padB + 14} textAnchor="middle" fontSize={11.5} fill="#8a93a3">
            {v.toFixed(0)}
          </text>
        </g>
      ))}
      <line x1={X(0)} x2={X(0)} y1={padT - 4} y2={H - padB} stroke="#9aa3b3" strokeWidth={1.4} />
      {rows.map((r, i) => {
        const y = padT + i * rowH + rowH / 2
        return (
          <g key={r.g}>
            <text x={padL - 8} y={y + 3.5} textAnchor="end" fontSize={13} fill="#4a5468" fontFamily="var(--mono)">
              {r.g}
            </text>
            {ci && (
              <line x1={X(ci[0])} x2={X(ci[1])} y1={y} y2={y} stroke="#0e7a65" strokeWidth={9} opacity={0.16} />
            )}
            <circle cx={X(r[loKey])} cy={y} r={5.2} fill="#16a085" />
            {hiKey && r[hiKey] != null && (
              <circle cx={X(r[hiKey])} cy={y} r={5.2} fill="#2980b9" />
            )}
          </g>
        )
      })}
      <text x={padL} y={H - 2} fontSize={12.5} fill="#8a93a3">{xlab}</text>
    </svg>
  )
}

export const Panel = ({ title, children, style }) => (
  <div className="panel" style={style}>
    {title ? <div className="ph">{title}</div> : null}
    {children}
  </div>
)

export const CIbadge = ({ mean, ci, unit = '' }) => (
  <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: 14 }}>
    <b>{mean > 0 ? '+' : ''}{mean.toFixed(2)}{unit}</b>{' '}
    <span style={{ color: '#8a93a3' }}>
      CI [{ci[0] > 0 ? '+' : ''}{ci[0].toFixed(2)}, {ci[1] > 0 ? '+' : ''}{ci[1].toFixed(2)}]
    </span>
  </span>
)