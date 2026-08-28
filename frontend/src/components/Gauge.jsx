export default function Gauge({ value=0.08, label='SAFE' }){
  // value 0..1 -> 0.08 like screenshot
  const pct = Math.max(0, Math.min(1, value))
  const angle = 270 * pct // 270deg arc
  const r = 46, c = 58, stroke = 8
  const circ = 2*Math.PI*r
  const dash = circ * 270/360
  const offset = dash - (dash * pct)
  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{width: 116, height:116}}>
        <svg width="116" height="116" viewBox="0 0 116 116" className="-rotate-135">
          <circle cx={c} cy={c} r={r} fill="none" stroke="#1e2323" strokeWidth={stroke} strokeLinecap="round" strokeDasharray={`${dash} ${circ}`} />
          <circle cx={c} cy={c} r={r} fill="none" stroke="#7cf000" strokeWidth={stroke} strokeLinecap="round" strokeDasharray={`${dash} ${circ}`} strokeDashoffset={offset} style={{filter:'drop-shadow(0 0 6px rgba(124,240,0,0.6))'}}/>
          <circle cx={c} cy={c} r={3.5} fill="#7cf000"/>
        </svg>
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center leading-none">
            <div className="mono text-[22px] font-semibold text-aegis-green" style={{textShadow:'0 0 10px rgba(124,240,0,0.6)'}}>{value.toFixed(2)}</div>
            <div className="mono text-[10px] tracking-[0.16em] text-red-500 mt-1">{label}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
