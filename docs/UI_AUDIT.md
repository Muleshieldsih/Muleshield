# UI audit — MuleShield AI console

Baseline commit `0294ccf`. Written before the refinement pass so the changes can
be checked against what was actually there.

The product concept, layout, dark theme, navigation, queue, money-flow graph,
location view and intervention actions are sound and are being kept. What
follows is what stops it reading as software an analyst uses daily.

---

## 1. Findings that are defects, not taste

These are wrong regardless of visual direction.

| # | Finding | Where |
|---|---|---|
| D1 | **A validation error is styled as a success.** Entering a bad phone number renders the message in the emerald success banner with a `CheckCircle2` tick. | `Interception.jsx:114`, `:329-334` |
| D2 | **The destructive action has no confirmation.** One click posts the freeze. Nothing asks. | `Interception.jsx:223-237` |
| D3 | **Two case-ID formats coexist.** Seed complaints carry a raw UUID; ingested ones carry `TKT-XXXXXXXX`. Two screens print the raw id instead of the formatted one, so the same case shows two identities. | `state.py:109` vs `:419`; `Interception.jsx:148`, `:309` |
| D4 | **Every graph edge animates permanently.** `animated: true` on all edges — marching dashes imply activity on a static historical record. | `ForensicGraph.jsx:90` |
| D5 | **The freeze log is write-only.** Freezes are recorded to `state.freeze_log` and never read back by any endpoint. There is no audit trail. | `state.py:69`, `freeze.py:63` |
| D6 | **Parallel transfers are silently dropped.** The graph endpoint de-duplicates edges by `src→dst`, so repeated transfers between the same pair collapse into one and disappear from the ledger the analyst sees. | `graph.py:215-218` |

## 2. Missing for an investigation product

| Capability | Status |
|---|---|
| Case status beyond `ACTIVE` | **None.** The field exists, is written twice, read never, and nothing mutates it. |
| Assignment to an analyst | **None.** The only person-identifier is a free-text `officer_id` on a freeze. |
| Case notes | **None.** |
| Audit log | **None** (see D5). |
| Transaction trail | **No endpoint.** 19 fields per transaction sit in memory; the only transport is `GraphResponse.edges`, which exposes 6 and drops the rest. |
| Pagination | **None.** 60 rows loaded, no page controls. |
| Toasts / confirmations | **None.** No dialog library; all feedback is a permanent inline banner. |
| Sorting by column | **None.** One fixed sort. |
| Filter by status / risk | **None.** Only search text and fraud type. |

Also loaded but never sent to the client: ATM `city`, `district`, `state`,
`opening_time`, `closing_time`, `cashout_risk_score` — opening hours in
particular are directly useful on a cash-out screen.

## 3. Typography

`.mono` is applied to **107 elements**, including prose. The pattern throughout
is that a container carries `mono` and everything inside inherits it, so
explanatory paragraphs, field labels, empty states and button text are all
monospaced.

Worst instances: the sentence *"sigmoid(Wh + b) from the trained GraphSAGE
classification head…"* (`ForensicGraph.jsx:270-273`), the whole error block
*"Complaints from the historical dataset always carry a ledger…"*
(`:191-194`), and the Leaflet basemap itself — `index.css:49` set **place
names** in JetBrains Mono.

Monospace should be reserved for case IDs, account numbers, amounts,
timestamps, coordinates and model versions.

## 4. Decorative effects with no operational purpose

| Effect | Where |
|---|---|
| `animate-pulse-dot` on static indicators | `Shell.jsx:133`, `TriageFeed.jsx:422`, `:592`, `Interception.jsx:157` |
| `animate-blink` on the 46px countdown | `Interception.jsx:66` |
| Neon glow on the freeze button | `Interception.jsx:229` — `shadow-[0_0_14px_rgba(255,59,59,0.3)]` |
| Glow on the logo tile and active nav | `Shell.jsx:120`, `:193` |
| Neon halo on map markers | `TacticalMap.jsx:17`, `:34` — inline `box-shadow` |
| Glow on terminal graph nodes | `ForensicGraph.jsx:81` |
| `backdrop-blur` scrims and floating chips | `TacticalMap.jsx:282-297`, `Interception.jsx:351`, `TriageFeed.jsx:251` |
| Basemap desaturated to zero | `index.css:53` — a map that cannot distinguish a road from a river |
| `drift` keyframe | `tailwind.config.js:33` — declared, never referenced, dead |

## 5. Terminology

| Current | Replacement |
|---|---|
| 1930 HELPLINE INTERDICTION CONSOLE | Fraud investigation console |
| TACTICAL GIS — PRIORITY SEARCH LOCATIONS | Cash-out locations |
| MONEY-FLOW DAG | Transaction trail |
| TACTICAL ACTIONS | Investigation |
| 1-CLICK EMERGENCY FREEZE | Freeze account |
| EMERGENCY LAW ENFORCEMENT ACTIONS | Intervention |
| FIELD PATROL DISPATCH / PCR VAN | Field dispatch |
| GRAPH CORPUS | (remove — it is an implementation statistic) |
| GRAPH ANOMALIES | Risk indicators |
| CYBER INTERCEPT ALERT | Interception alert |
| ARMED / STANDBY | Ready / No case selected |

## 6. Model metrics in the wrong place

The sidebar carries `Top-5 containment`, `Search zone`, `Countdown MAE` and
`Mule F1` on every screen, including while investigating a single case. These
describe the detection system's behaviour across a test set; they say nothing
about the case on screen, and placing them beside it invites exactly that
misreading. They belong on their own screen.

## 7. What is already right and is being kept

- The dark palette, the green identity, and the panel chrome.
- Information density — this is a desktop analyst tool and should stay dense.
- The plain-language case brief.
- The severity model and its honesty about the golden hour.
- Keyboard focus rings, `prefers-reduced-motion` handling, degraded states that
  say what is wrong rather than substituting fixtures.

---

## Sequence

1. **Visual system** — typography, effects, colour discipline, terminology.
2. **Case queue** — status model, filters, sorting, pagination, ID format.
3. **Investigation** — transaction trail endpoint and table, timeline, entity panel.
4. **Locations** — expose the ATM fields already loaded.
5. **Intervention and audit** — confirmation, audit log, toasts.
6. **Regression pass** — smoke sweep and the Python suite.
