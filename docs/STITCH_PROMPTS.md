# Stitch AI prompts — MuleShield AI console

Black theme, security-operations aesthetic. Colour is used **functionally only**
— severity, status, and one accent — never as decoration. Palette tokens match
`frontend/tailwind.config.js` so the design and the code stay in step.

How to use: paste **Prompt 0** first on its own to lock the design system, then
generate each screen with its own prompt in the same project. App type: **Web**.

---

## Prompt 0 — Design system (paste first)

> Design a **black-theme desktop web console** for **MuleShield AI**, a
> cyber-crime investigation tool used by Indian police cyber cells and bank fraud
> officers to trace stolen money through mule bank accounts and predict which ATM
> it will be withdrawn from, inside a 60-minute window.
>
> Aesthetic: a **serious security-operations terminal** — the look of a SOC
> console, a packet analyser, a forensic tool. Sharp, technical, dense, quiet.
> **Not** neon cyberpunk: no glows, no scanlines, no pulsing dots, no blinking,
> no blur, no gradients, no drop shadows, no matrix rain.
>
> **Surfaces (the interface is ~95% neutral):**
> - Page `#000000`. Surface `#0A0A0A`. Panel `#111111`. Raised `#171717`.
> - Hairlines `#1F1F1F`, dividers `#2A2A2A`, focus outline `#3D3D3D`.
> - Text: primary `#F5F5F5`, secondary `#A1A1A1`, tertiary `#6E6E6E`, disabled `#454545`.
>
> **Colour is functional and rationed.** It appears only on small elements that
> report state — never on backgrounds, panels, headers, large fills or as
> ornament:
> - Accent `#7CF000` (signal green) — the active nav item, the primary action,
>   and the connected-link indicator. Nothing else.
> - `#FF3B3B` critical · `#FF8C42` warning · `#7CF000` monitoring · `#58A6FF`
>   informational — used **only** to report status, at small sizes: a 3px rule, a
>   pill border, a glyph, a single figure.
> - Any coloured element also carries an explicit uppercase text label
>   (`CRITICAL` / `WARNING` / `MONITOR`) so severity survives a projector, a bad
>   monitor and colour-blind viewers.
>
> **Typography:**
> - **JetBrains Mono** for all data: case IDs, account numbers, ₹ amounts,
>   timestamps, countdowns, coordinates, percentages, status labels, table
>   headers, and every micro-label (uppercase, 10px, letter-spacing 0.08em).
> - **Inter** for prose, descriptions and button text only. Never set a sentence
>   in monospace.
> - Scale: 10px uppercase labels, 12–13px body, 15px section titles, 32–46px only
>   for a live countdown. Line height 1.4. Tabular figures on every number.
>
> **Chrome:**
> - Fixed 48px top bar, fixed 210px left sidebar, dense scrollable main panel,
>   24px status footer. 1440×900 desktop canvas.
> - Corners are **2px or square** — nothing rounded or soft.
> - Every panel is a 1px-bordered rectangle with a 30px header row: an uppercase
>   mono title on the left, a secondary figure in tertiary grey on the right.
> - Dense: 8–12px padding, 32px table rows, tables preferred over cards.
> - Buttons are flat 1px-outlined rectangles with no fill and no shadow. The
>   primary action takes a green border and green text. Destructive actions take
>   a red border and a `⚠` glyph — outlined, never filled red.
>
> **Top bar:** a shield glyph in a green-outlined square tile, `MULESHIELD` in
> mono uppercase with the subtitle "Fraud investigation console · 1930 / I4C" in
> 11px grey; on the right a status readout `LINK ● ACTIVE` with a small green dot
> and the active case reference `1930-482910` in mono.
>
> **Left sidebar nav** — uppercase mono labels with a grey one-line description
> each: CASES (Queue and triage) · TRAIL (Fund movement) · LOCATIONS (Cash-out
> points) · INTERVENTION (Freeze and escalate) · MODEL (Detection accuracy). The
> active item has a green left rule and white text; the rest are tertiary grey.
> Below the nav, a scrollable compact case queue. Footer: `SIH26184 · MHA / I4C`.

---

## Prompt 1 — Cases (triage queue)

> Screen: **CASES** — the incoming complaint queue from India's 1930 cyber-crime
> helpline.
>
> Filter bar across the top: a search field with a `/` hint glyph, dropdowns for
> fraud type (UPI Fraud, Digital Arrest, Job Scam, Investment Scam, KYC Fraud),
> status (New, Assigned, Investigating, Frozen, Closed) and severity, then a
> right-aligned outlined `+ LOG COMPLAINT` button.
>
> A **dense sortable table**, ~14 rows visible, 32px rows, uppercase mono column
> headers with sort carets. Columns: Severity (a 3px left rule — red `#FF3B3B`
> critical, amber `#FF8C42` warning, grey monitoring — plus a `▲ ■ ·` glyph and
> the uppercase label), Case ref (`1930-482910`), Victim, Bank, Fraud type,
> Amount (right-aligned `₹1,20,000`), City, Elapsed (`08 MIN`), Status (a
> 1px-outlined pill, uppercase mono, grey border), Assignee. Hover lifts the row
> to `#111111`; the selected row carries a 2px green left edge and near-white text.
>
> Pagination beneath in mono: `◀ 1–14 of 60 ▶`.
>
> Right rail, 360px, showing the selected case: victim, bank, amount stolen,
> fraud type, city and complaint time as a mono key/value table with hairline
> dividers; a plain-English case brief paragraph in Inter; a **large mono
> countdown** of golden-hour time remaining, white when ample and red under ten
> minutes, with a thin segmented progress bar (filled segments white, empty
> `#2A2A2A`); then three stacked outlined buttons — `VIEW TRAIL`,
> `VIEW LOCATIONS`, `OPEN INTERVENTION`.
>
> Above the table, four metric tiles: Active cases, Critical now, Total at risk
> (₹), Median response. 10px uppercase grey label, large white mono value, no
> icons. Only "Critical now" is coloured, and only when non-zero.

---

## Prompt 2 — Transaction trail (money-flow graph)

> Screen: **TRAIL** — a directed node-link graph showing stolen money moving from
> a victim account through layers of mule accounts.
>
> Main canvas: a left-to-right DAG on pure black with a faint `#171717` dot grid.
> Nodes are square-cornered rectangles with a 1px border carrying an account
> number in mono, the bank at 10px grey, and the amount received. Node types are
> distinguished by border colour **and** an explicit label:
> - **VICTIM** (hop 0) — blue `#58A6FF` border, transparent fill, label `VICTIM`.
> - **MULE L1 / L2** — amber `#FF8C42` border, `#111111` fill, label `MULE · HOP 2`.
> - **TERMINAL** — red `#FF3B3B` 2px border, `#171717` fill, label `▲ TERMINAL`.
>
> Edges are thin static grey curves labelled with the transferred amount and the
> minutes elapsed between hops; edge thickness scales with amount. No animated
> dashes, no glow on any node. Zoom / fit / re-layout controls bottom-left as
> outlined square icon buttons; a compact legend bottom-right in mono.
>
> Right rail, 380px, two stacked panels:
> 1. **RISK INDICATORS** — a ranked list of the highest-risk accounts, each with
>    a mono score `0.00–1.00` and a thin horizontal bar against a `#2A2A2A`
>    track, plus one line of Inter prose explaining what the score is.
> 2. **SELECTED ACCOUNT** — account number, bank, hop depth, total in / out,
>    first and last seen, then a small mono table of that account's individual
>    transactions (time, counterparty, amount, channel).
>
> Header strip above the canvas, all mono uppercase, separated by `·`: case ref,
> hop count, node count, total traced, and `RECONSTRUCTED 184 MS`.

---

## Prompt 3 — Locations (cash-out prediction map)

> Screen: **LOCATIONS** — where the model predicts the cash will be withdrawn.
>
> Left two-thirds: a **dark map of Delhi NCR** with legible muted roads and grey
> street labels — dimmed and low-saturation, but not stripped to pure grey; a
> road must still read differently from a river. Overlays: the predicted search
> zone as a 1px dashed green circle with a 4% green fill and its radius labelled
> in mono along the edge; three ATM candidates as square 1px-outlined markers
> carrying rank numerals `1 2 3` — rank 1 filled red `#FF3B3B` with black
> numerals, rank 2 amber `#FF8C42`, rank 3 grey outline only. Flat markers: no
> pins, no glow, no halo.
>
> A small floating outlined panel on the map gives the zone summary in mono:
> centroid coordinates, radius in km, and `12 OF 1,847 ATMS IN ZONE`.
>
> Right third: **RANKED CANDIDATES** as three stacked outlined cards, the first
> with a 2px red left rule. Each card: large mono rank numeral, ATM ID
> (`ATM-SBI-004`), bank, street address, district, confidence percentage with a
> thin bar, distance from the last known mule location, opening hours, a
> surveillance-risk label in uppercase mono, and an outlined `DISPATCH HERE`
> button.
>
> Below the cards, a **COUNTDOWN** panel: a 46px mono figure `27:43` in white,
> turning red under ten minutes; the label "estimated time to withdrawal" in grey
> Inter; and beneath it the prediction band `q05 19 MIN — q95 41 MIN` in small mono.

---

## Prompt 4 — Intervention

> Screen: **INTERVENTION** — the actions an officer takes on a live case.
>
> Left column: a **case summary** strip (case ref, victim, amount, target ATM,
> countdown — all mono, separated by hairline dividers), then two action panels.
>
> 1. **FREEZE ACCOUNT** — target mule account number and bank in mono, an officer
>    ID field, a reason dropdown, and a **red-outlined** button labelled
>    `⚠ FREEZE ACCOUNT` — outlined, never filled. It opens a **confirmation
>    modal**: a small centred square-cornered panel on a 70%-black scrim, titled
>    `CONFIRM FREEZE`, the account number large in mono, one line of Inter prose
>    saying the effect is immediate and reversible only by the issuing bank, then
>    a grey-outlined `CANCEL` and a red-outlined `⚠ CONFIRM`.
> 2. **FIELD DISPATCH** — nearest patrol unit with distance and ETA, a phone
>    field with inline validation, a message preview inside a bordered mono block
>    on `#0A0A0A`, and an outlined `SEND ALERT` button.
>
> Right column: **CASE ACTIVITY** — a vertical audit timeline, newest first, each
> entry a mono timestamp, actor, an uppercase action (`COMPLAINT INGESTED`,
> `PREDICTION GENERATED`, `ACCOUNT FROZEN`, `DISPATCH SENT`) and a one-line grey
> detail. A 1px vertical hairline connects entries; each node is a small hollow
> square, filled green only for the most recent.
>
> Toasts appear bottom-right as small square-cornered rectangles with a close
> glyph. Success has a green left edge and a `✓` prefix; failure has a red left
> edge, a `✕` prefix and bolder text. A validation error must never be styled as
> a success.

---

## Prompt 5 — Model performance

> Screen: **MODEL** — how the detection system behaves across its test set, kept
> deliberately separate from any single case.
>
> Top: four metric tiles, each a 10px uppercase grey label, a large white mono
> value, and a smaller tertiary-grey comparison line — Top-5 containment (71.4%,
> "search space reduced 99.7%"), Search zone (8.2 km median radius), Countdown
> MAE (6.4 min, "vs 14.1 min baseline"), Mule detection F1 (0.9214, "vs 0.8630
> XGBoost baseline"). The values stay white — these are measurements, not alerts.
>
> Below, two charts side by side on pure black with 1px `#2A2A2A` axes, `#171717`
> grid lines and mono tick labels — no 3D, no shadows, no gradient fills:
> - A **Top-K containment curve** drawn as a single 1.5px green line, with the
>   baseline as a 1px dashed grey line and a small mono legend.
> - A **confusion matrix** as a 2×2 grid of square cells with mono counts, cells
>   shaded by magnitude from `#0A0A0A` up to saturated green.
>
> Beneath them a **model comparison table**: model name, precision, recall, F1,
> inference latency — every figure mono and right-aligned, the shipped model's
> row marked with a 2px green left rule and near-white text.
>
> Close with a note panel of two or three sentences in grey Inter prose stating
> plainly what these numbers do and do not claim.

---

## Follow-up prompts for iteration

- "Keep colour functional only — remove any colour that is not reporting status. Backgrounds, panels and headers stay neutral."
- "Remove every glow, shadow, gradient and blur. Keep only 1px hairline borders."
- "Square the corners — 2px maximum radius, nothing rounded."
- "Increase density: 8px padding, 32px row height, fit four more rows on screen."
- "Set all data, IDs, timestamps and micro-labels in JetBrains Mono with tabular figures; use Inter only for sentences."
- "Show the empty state: no case selected, explaining what to do next."
- "Show the error state: the backend is unreachable."
