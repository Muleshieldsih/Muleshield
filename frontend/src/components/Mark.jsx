/**
 * The MuleShield mark.
 *
 * A radiant crosshair rather than a shield. A shield says defence — hold the
 * line, keep them out — and that is not what this system does. Everything
 * upstream of it has already failed by the time a complaint arrives: the money
 * is gone and moving. What this product claims is INTERCEPTION, a point fixed
 * on a map with minutes left on it, and a reticle says that in one glance.
 *
 * The bright pip at the centre is the only fill in the mark. It is the machine
 * the model picked.
 *
 * Three sets of proportions, because one does not survive the range. At 16px in
 * the topbar a stroke drawn to look delicate at 240px renders at under half a
 * pixel and disappears into grey mush; at 240px a stroke heavy enough to read
 * at 16px looks like a road sign. The switches are at 32px, where the app's
 * chrome ends, and at 96px, where the mark stops being an icon and becomes the
 * subject of the screen -- on the sign-in plate it is drawn finest of all, with
 * the diagonals dropped back to 62% so the cardinals carry it alone.
 */
export default function Mark({ size = 24, className = '', pip = '#fafafa' }) {
  const small = size <= 32
  const display = size > 96

  // Geometry is expressed in a 24-unit grid to sit on the same rhythm as the
  // lucide icons used everywhere else in the console.
  const arm = small
    ? { stroke: 2.1, inner: 7.4, outer: 1.4, diag: { stroke: 1.6, from: 5.1, to: 8.3, opacity: 0.9 }, r: 1.15 }
    : display
      ? { stroke: 0.48, inner: 7.84, outer: 1.28, diag: { stroke: 0.36, from: 6, to: 9.2, opacity: 0.62 }, r: 0.28 }
      : { stroke: 1.15, inner: 6.6, outer: 0.9, diag: { stroke: 0.95, from: 4.6, to: 8.1, opacity: 0.9 }, r: 0.62 }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      role="img"
      aria-label="MuleShield AI"
    >
      {/* cardinal arms — flat caps, so the mark reads as struck rather than drawn */}
      <g stroke="currentColor" strokeWidth={arm.stroke} strokeLinecap="butt">
        <path d={`M12 ${arm.outer}V${arm.inner}`} />
        <path d={`M12 ${24 - arm.inner}V${24 - arm.outer}`} />
        <path d={`M${arm.outer} 12H${arm.inner}`} />
        <path d={`M${24 - arm.inner} 12H${24 - arm.outer}`} />
      </g>

      {/* diagonals — shorter and lighter, so they read as bearing ticks and do
          not compete with the cardinals at small sizes */}
      <g stroke="currentColor" strokeWidth={arm.diag.stroke} strokeLinecap="butt" opacity={arm.diag.opacity}>
        <path d={`M${arm.diag.from} ${arm.diag.from}L${arm.diag.to} ${arm.diag.to}`} />
        <path d={`M${24 - arm.diag.from} ${arm.diag.from}L${24 - arm.diag.to} ${arm.diag.to}`} />
        <path d={`M${arm.diag.from} ${24 - arm.diag.from}L${arm.diag.to} ${24 - arm.diag.to}`} />
        <path d={`M${24 - arm.diag.from} ${24 - arm.diag.from}L${24 - arm.diag.to} ${24 - arm.diag.to}`} />
      </g>

      <circle cx="12" cy="12" r={arm.r} fill={pip} stroke="none" />
    </svg>
  )
}
