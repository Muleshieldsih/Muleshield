# Stitch AI — MuleShield AI console, single combined prompt

Everything in one paste: design system plus six screens. The per-screen version
is in `STITCH_PROMPTS.md` if the output comes back too generic.

Influences folded in: the Aegis SOC dashboard (live event stream, embedded
terminal, one hero index with a state word, approval cards, sharp edges) and
Allel (numbered 01–05 pipeline, uppercase micro-role labels, a big-numeral stat
trio, a priority-ranked shift brief, editorial calm on overview surfaces).

---

Design a **black-theme desktop web console** for **MuleShield AI**, a cyber-crime
investigation tool used by Indian police cyber cells and bank fraud officers. It
traces stolen money through mule bank accounts and predicts which ATM it will be
withdrawn from, inside a 60-minute window. Generate **six connected screens**.

**AESTHETIC** — a serious security-operations terminal: a SOC console, a packet
analyser, a forensic tool. Sharp, technical, dense, quiet. Built for an operator
who stares at it for eight hours. Two registers, deliberately: **operational
screens are dense** (tables, hairlines, 32px rows), while **overview surfaces
breathe** (generous whitespace, large numerals, editorial calm) — the contrast
between them is the design. NOT neon cyberpunk: no glows, no scanlines, no pulsing
dots, no blinking, no blur, no gradients, no drop shadows, no matrix rain.

**SURFACES** (the interface is ~95% neutral) — Page `#000000`, surface `#0A0A0A`,
panel `#111111`, raised `#171717`. Hairlines `#1F1F1F`, dividers `#2A2A2A`, focus
`#3D3D3D`. Text: primary `#F5F5F5`, secondary `#A1A1A1`, tertiary `#6E6E6E`,
disabled `#454545`.

**COLOUR IS FUNCTIONAL AND RATIONED** — only on small elements reporting state,
never on backgrounds, panels, headers, large fills or as ornament. Accent
`#7CF000` signal green for the active nav item, the primary action and the
connected indicator, nothing else. Status: `#FF3B3B` critical, `#FF8C42` warning,
`#7CF000` monitoring, `#58A6FF` informational — at small sizes only: a 3px rule, a
pill border, a glyph, a single figure. Every coloured element also carries an
uppercase text label so severity survives a projector and a colour-blind viewer.

**TYPOGRAPHY** — JetBrains Mono for all data: case IDs, account numbers, ₹
amounts, timestamps, countdowns, coordinates, percentages, status labels, table
headers, terminal text, and every micro-label (uppercase, 10px, letter-spacing
0.08em). Inter for prose and button text only — never a sentence in monospace.
Tabular figures on every number. Scale: 10px labels, 12–13px body, 15px section
titles, 32–46px for a live countdown or a hero index, 64px for an overview
numeral. Line height 1.4.

**RECURRING MOTIFS**, used across screens so it reads as one system:
- **Numbered pipeline** — the five stages `01 INGEST · 02 TRACE · 03 SCORE ·
  04 PREDICT · 05 INTERVENE` rendered as uppercase mono with the two-digit numeral
  in tertiary grey. Wherever a case's progress is shown, completed stages are
  white, the current stage green, pending stages `#454545`.
- **Uppercase micro-role label** above every named entity — `VICTIM`,
  `MULE · HOP 2`, `TERMINAL MULE`, `PATROL UNIT`, `CASE OFFICER`, in 10px mono
  tertiary grey with letter-spacing.
- **Hero index tile** — one headline figure with a state word beside it, e.g.
  `MULE RISK INDEX  0.84  ▲ CRITICAL`: a 46px mono value, the state word in a
  1px-outlined uppercase pill. Used once per screen at most.
- **Live event stream** — a monospace ledger, newest at top, each line a
  timestamp, an uppercase event tag and a terse detail, older lines fading through
  the grey ramp toward the bottom of the panel.
- **Embedded command line** — a collapsible strip with the prompt
  `officer@muleshield:~$` in green mono, a blinking-free caret, and grey hint text
  showing available commands (`trace <case>`, `predict <case>`, `freeze <account>`).

**CHROME** — 1440×900. Fixed 48px top bar, fixed 210px left sidebar, dense
scrollable main panel, 24px status footer. Corners 2px or square, nothing soft.
Every panel is a 1px-bordered rectangle with a 30px header row: uppercase mono
title left, secondary figure in tertiary grey right. Dense: 8–12px padding, 32px
table rows, tables preferred over cards. Buttons are flat 1px outlines, no fill, no
shadow; primary takes a green border, destructive a red border and a `⚠` glyph —
outlined, never filled.

**PERSISTENT SHELL, on every screen.** Top bar: shield glyph in a green-outlined
square tile, `MULESHIELD` in mono uppercase, subtitle "Fraud investigation console
· 1930 / I4C" in 11px grey; centre-right the pipeline stage strip for the active
case; right side `LINK ● ACTIVE` with a small green dot and the active case
reference `1930-482910` in mono. Left sidebar: uppercase mono nav with a grey
one-line description each — OVERVIEW (Shift brief), CASES (Queue and triage),
TRAIL (Fund movement), LOCATIONS (Cash-out points), INTERVENTION (Freeze and
escalate), MODEL (Detection accuracy); the active item takes a green left rule and
white text, the rest tertiary grey; below it a scrollable compact case queue;
footer `SIH26184 · MHA / I4C`.

---

**SCREEN 0 — OVERVIEW.** The whole operation on one screen, and the one place the
layout is allowed to breathe. A page title in Inter, "Your shift, ranked by what
is about to happen", with a mono timestamp beneath: `SHIFT BRIEF — 07:00 IST`.
Below it a stat trio in a wide-spaced row, each a 64px white mono numeral over a
10px uppercase grey caption: `14 ACTIVE CASES`, `3 INSIDE GOLDEN HOUR`,
`₹42.6L AT RISK NOW`. Then the numbered pipeline `01 INGEST · 02 TRACE · 03 SCORE
· 04 PREDICT · 05 INTERVENE` drawn across the full width as five hairline-separated
cells, each with the stage name, a one-line grey Inter description, and a mono
count of cases currently at that stage. Beneath, two columns: on the left a
**PRIORITY BRIEF** — a ranked list of the top five cases, each row a rank numeral,
case ref, a one-sentence plain-English summary in Inter ("₹1.2L from a Delhi
victim reached a terminal mule 6 minutes ago; withdrawal predicted within 28
minutes"), a countdown in mono, and a severity rule; on the right the **LIVE
STREAM** — a monospace event ledger, newest at top, lines like
`07:04:12  INGEST   1930-482910  ₹1,20,000  DELHI`,
`07:03:58  PREDICT  1930-771204  ATM-SBI-004  91.2%`,
`07:02:31  FREEZE   XXXXXXXX4471  EXECUTED`, older lines fading toward the bottom.
At the foot of the screen the collapsible command line strip.

**SCREEN 1 — CASES.** The incoming complaint queue from India's 1930 cyber-crime
helpline. Filter bar on top: search field with a `/` hint glyph, dropdowns for
fraud type (UPI Fraud, Digital Arrest, Job Scam, Investment Scam, KYC Fraud),
status (New, Assigned, Investigating, Frozen, Closed) and severity, then a
right-aligned outlined `+ LOG COMPLAINT`. Four metric tiles above the table —
Active cases, Critical now, Total at risk (₹), Median response — 10px uppercase
grey label over a large white mono value, no icons, only "Critical now" coloured
and only when non-zero. Then a dense sortable table, ~14 rows at 32px, uppercase
mono headers with sort carets: Severity (3px left rule, red/amber/grey, plus a
`▲ ■ ·` glyph and an uppercase label), Case ref `1930-482910`, Victim, Bank, Fraud
type, Amount right-aligned `₹1,20,000`, City, Elapsed `08 MIN`, Stage (the
pipeline position, e.g. `04 PREDICT`), Status as a 1px-outlined mono pill,
Assignee. Hover lifts the row to `#111111`; the selected row takes a 2px green left
edge. Pagination beneath: `◀ 1–14 of 60 ▶`. Right rail 360px for the selected
case: the `VICTIM` micro-label above the name, a mono key/value table with hairline
dividers (bank, amount stolen, fraud type, city, complaint time), a plain-English
case brief paragraph in Inter, a large mono golden-hour countdown — white when
ample, red under ten minutes — with a thin segmented progress bar (filled white,
empty `#2A2A2A`), then three stacked outlined buttons `VIEW TRAIL`,
`VIEW LOCATIONS`, `OPEN INTERVENTION`.

**SCREEN 2 — TRAIL.** A directed node-link graph of stolen money moving from a
victim account through layers of mule accounts. Left-to-right DAG on pure black
with a faint `#171717` dot grid. Nodes are square-cornered rectangles, 1px border,
each carrying its uppercase micro-role label, an account number in mono, the bank
at 10px grey, and the amount received; distinguished by border colour AND the
label — VICTIM (hop 0, blue `#58A6FF` border, transparent fill), MULE · HOP 2
(amber `#FF8C42` border, `#111111` fill), ▲ TERMINAL MULE (red `#FF3B3B` 2px
border, `#171717` fill). Edges are thin static grey curves labelled with the
transferred amount and minutes elapsed between hops, thickness scaling with amount
— no animated dashes, no glow on any node. Zoom/fit/re-layout controls bottom-left
as outlined square icon buttons, compact mono legend bottom-right. Header strip
above the canvas, mono uppercase separated by `·`: case ref, hop count, node count,
total traced, `RECONSTRUCTED 184 MS`. Right rail 380px, two panels: a hero index
tile `MULE RISK INDEX 0.84 ▲ CRITICAL` above **RISK INDICATORS** (ranked accounts,
each a mono score `0.00–1.00` with a thin bar on a `#2A2A2A` track, plus one line
of Inter prose saying what the score is), then **SELECTED ACCOUNT** (account
number, bank, hop depth, total in/out, first and last seen, and a small mono table
of that account's transactions — time, counterparty, amount, channel).

**SCREEN 3 — LOCATIONS.** Where the model predicts the cash will be withdrawn.
Left two-thirds: a dark map of Delhi NCR, dimmed and low-saturation with legible
grey street labels — a road must still read differently from a river. Overlays: the
predicted search zone as a 1px dashed green circle with a 4% green fill and its
radius labelled in mono along the edge; three ATM candidates as square
1px-outlined markers carrying rank numerals `1 2 3` — rank 1 filled red with black
numerals, rank 2 amber, rank 3 grey outline only. Flat markers: no pins, no glow,
no halo. A small floating outlined panel on the map gives the zone summary in mono:
centroid coordinates, radius in km, `12 OF 1,847 ATMS IN ZONE`. Right third:
**RANKED CANDIDATES** as three stacked outlined cards, the first with a 2px red
left rule — each with a large mono rank numeral, ATM ID `ATM-SBI-004`, bank, street
address, district, confidence percentage with a thin bar, distance from the last
known mule location, opening hours, an uppercase mono surveillance-risk label, and
an outlined `DISPATCH HERE`. Below them a COUNTDOWN panel: a 46px mono `27:43` in
white turning red under ten minutes, the label "estimated time to withdrawal" in
grey Inter, and the prediction band `q05 19 MIN — q95 41 MIN` in small mono.

**SCREEN 4 — INTERVENTION.** The actions an officer takes on a live case. Left
column: a case summary strip (case ref, victim, amount, target ATM, countdown, all
mono, hairline dividers), then an **APPROVAL QUEUE** of stacked action cards, each
one a proposed intervention awaiting a human decision — an uppercase action tag, a
one-sentence Inter justification of why the system is proposing it, the target in
mono, a confidence figure, and two outlined buttons `APPROVE` / `DISMISS`. Beneath
it two panels. FREEZE ACCOUNT — target mule account number and bank in mono, an
officer ID field, a reason dropdown, and a red-outlined `⚠ FREEZE ACCOUNT` button,
never filled; it opens a confirmation modal, a small centred square-cornered panel
on a 70%-black scrim titled `CONFIRM FREEZE`, the account number large in mono,
one line of Inter prose saying the effect is immediate and reversible only by the
issuing bank, then a grey-outlined `CANCEL` and a red-outlined `⚠ CONFIRM`. FIELD
DISPATCH — the `PATROL UNIT` micro-label above the nearest unit with distance and
ETA, a phone field with inline validation, a message preview in a bordered mono
block on `#0A0A0A`, an outlined `SEND ALERT`. Right column: **CASE ACTIVITY**, a
vertical audit timeline newest-first, each entry a mono timestamp, actor, an
uppercase action (`COMPLAINT INGESTED`, `PREDICTION GENERATED`, `ACCOUNT FROZEN`,
`DISPATCH SENT`) and a one-line grey detail, connected by a 1px vertical hairline
with a small hollow square per node, filled green only for the most recent. At the
foot, the command line strip with `officer@muleshield:~$ freeze XXXXXXXX4471`
entered and its response beneath in grey mono. Toasts bottom-right as small
square-cornered rectangles with a close glyph: success has a green left edge and a
`✓` prefix, failure a red left edge, a `✕` prefix and bolder text — a validation
error must never be styled as a success.

**SCREEN 5 — MODEL.** How the detection system behaves across its test set, kept
deliberately separate from any single case. This screen uses the roomier register:
a page title in Inter and four metric tiles in a wide-spaced row, each a 10px
uppercase grey label over a large white mono value with a smaller tertiary-grey
comparison line — Top-5 containment 71.4% ("search space reduced 99.7%"), Search
zone 8.2 km median radius, Countdown MAE 6.4 min ("vs 14.1 min baseline"), Mule
detection F1 0.9214 ("vs 0.8630 XGBoost baseline"); the values stay white, these
are measurements not alerts. Below, two charts side by side on pure black with 1px
`#2A2A2A` axes, `#171717` grid lines and mono ticks — no 3D, no shadows, no
gradient fills: a Top-K containment curve as a single 1.5px green line with the
baseline as a 1px dashed grey line and a small mono legend, and a confusion matrix
as a 2×2 grid of square cells with mono counts shaded by magnitude from `#0A0A0A`
up to saturated green. Then a model comparison table — model name, precision,
recall, F1, inference latency, every figure mono and right-aligned, the shipped
model's row marked with a 2px green left rule. Close with a note panel of two or
three sentences in grey Inter prose stating plainly what these numbers do and do
not claim.
