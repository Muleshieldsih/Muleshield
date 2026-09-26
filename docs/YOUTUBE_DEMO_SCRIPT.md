# MuleShield AI — Official 3-Minute SIH YouTube Demo Video Script

**Problem Statement ID:** SIH26184  
**Project Name:** MuleShield AI  
**Theme:** Blockchain & Cybersecurity  
**Video Target Duration:** Exactly 3 Minutes (180 Seconds)  
**Target Speaking Pace:** ~135–140 words per minute (~410 spoken words total)

---

## 📋 Video Structure Overview

| Timestamp | Phase | Screen Focus | Key Message |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:35** (35s) | **The Problem & Operational Void** | Slide 1 / Real Crime Context | Digital trails vanish at ATMs; no tool forecasts cash-out locations. |
| **0:35 – 1:05** (30s) | **The Solution & Core Innovation** | System Architecture / Slide 2 | Graph BFS + GraphSAGE + Nested Logit + Survival Countdown. |
| **1:05 – 2:25** (80s) | **Live Working Prototype Demo** | Live Web Console (5 Key Views) | Ingestion $\to$ Graph Trail $\to$ Map & Countdown $\to$ Dispatch & Interlock. |
| **2:25 – 3:00** (35s) | **Impact, Feasibility & Wrap-Up** | Model Matrix / Section 63 Proof | 7.5 ms switch latency, 87.3% zone containment, ready for pilot. |

---

## 🎬 Master Storyboard & Voiceover (Time-Stamped)

### Part 1: The Problem & The Golden Hour Void (0:00 – 0:35)
* **Visual on Screen:**
  - *[0:00 – 0:10]* Start on Title Slide or high-impact intro graphic showing the 1930 Cyber Helpline & stolen fund dispersal flow.
  - *[0:10 – 0:25]* Transition to a diagram of money moving through 3–4 bank accounts, ending with an ATM icon flashing in red.
  - *[0:25 – 0:35]* Show the limitation: Once cash leaves the dispenser, the money is gone forever.
* **Voiceover:**
  > "Every single day across India, thousands of citizens fall victim to cyber-financial fraud. When a victim dials the national 1930 helpline, money has already been laundered through multiple mule accounts within minutes.
  > 
  > Current systems like CFCFRMS can freeze accounts if money is still in the bank, and tools like Pratibimb only show where crimes happened in the past. But once a syndicate runner physically withdraws paper currency from an ATM, the digital trail goes cold.
  > 
  > There is an operational void: **No existing system tells police WHICH physical ATM to intercept, and WHEN.**"

---

### Part 2: Introducing MuleShield AI (0:35 – 1:05)
* **Visual on Screen:**
  - *[0:35 – 0:45]* Reveal **MuleShield AI** logo and high-level pipeline diagram (Graph $\to$ Nested Logit $\to$ Countdown $\to$ Switch).
  - *[0:45 – 1:05]* Highlight key metrics callout banner: **87.3% Top-5 Zone Containment**, **40.2 min warning lead time**, **7.5 ms switch latency**.
* **Voiceover:**
  > "To solve this, we built **MuleShield AI** — the first predictive analytics framework that forecasts physical cash withdrawal locations in advance, enabling proactive law enforcement interdiction during the golden hour.
  > 
  > MuleShield AI integrates four breakthrough innovations:
  > 1. **Deterministic Multi-Hop Graph Traversal** to track money flow in real-time.
  > 2. **Inductive GraphSAGE Embeddings** achieving an F1-score of 0.905 for mule account detection.
  > 3. **Nested Logit Discrete Choice Modeling**, relaxing the IIA assumption to achieve 87.3% physical search zone containment.
  > 4. And a **Dual-Regime Survival Countdown**, providing a median 40 minutes of operational lead time before cash dispensing."

---

### Part 3: Live Working Prototype Walkthrough (1:05 – 2:25)
* **Visual on Screen & Actions:**
  - *[1:05 – 1:20]* **Screen 1: Case Queue** (`01-case-queue.png` / Live UI)
    - Show the incoming live complaint stream.
    - Click into a fresh case: `CMP-2024-88412` (Amount: ₹2,40,000). Show instant 5.2 ms inference.
  - *[1:20 – 1:40]* **Screen 2: Multi-Hop Transaction Trail** (`02-transaction-trail.png`)
    - Switch to the interactive graph viewer. Zoom in on victim $\to$ layer 1 $\to$ layer 2 $\to$ terminal mule node.
    - Show the account highlighted with GraphSAGE risk score `0.942`.
  - *[1:40 – 2:00]* **Screen 3: Spatial Map & Withdrawal Countdown** (`03-cash-out-locations.png`)
    - Pan across the interactive Leaflet GIS map.
    - Show the 5 highlighted ATMs ranked by choice probability.
    - Point cursor to the active withdrawal countdown timer: **`34m 12s remaining`**.
  - *[2:00 – 2:15]* **Screen 4: Switch Interlock & Dispatch Console** (`04-intervention.png`)
    - Click **'Dispatch Patrol'** to send coordinate alerts to nearest field units.
    - Click **'Trigger Switch Interlock'** — show ISO 8583 Field 39 deceptive denial simulation (injecting Code `05` / `51` to quietly delay the runner on-site without sounding an alarm).
  - *[2:15 – 2:25]* **Screen 5: Legal Evidence & Hotspot Surface** (`06-risk-heatmap.png` & `09-evidence-certificate.png`)
    - Show the 226-cell national Forward Cash-Out Surface (6.0× PAI lift over historical heatmaps).
    - Briefly show the cryptographic SHA-256 certificate compliant with **BSA 2023 Section 63**.
* **Voiceover:**
  > "Let's see MuleShield AI in action.
  > 
  > Here is our live operations console. As a complaint is ingested, our deterministic BFS engine immediately traces the multi-hop fund dispersal across banking boundaries. 
  > 
  > In this transaction graph, our GraphSAGE model instantly pinpoints the terminal cash-out mule holding the remaining funds.
  > 
  > Moving to the spatial intelligence view, MuleShield AI narrows 1,000 candidate terminals down to just 5 high-probability ATMs on an interactive GIS map. Alongside it, our survival model starts a live tactical countdown — here giving police 34 minutes of advance warning.
  > 
  > With a single click, the duty officer dispatches ground patrol units to the top search zone. Simultaneously, our switch integration architecture proposes an ISO 8583 Field 39 deceptive response code — such as 'Do Not Honor' — suppressing the cash dispenser and buying crucial minutes for field interception.
  > 
  > Every action is immutably anchored in a SHA-256 ledger, generating a legally admissible certificate under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023."

---

### Part 4: Feasibility, Viability & Conclusion (2:25 – 3:00)
* **Visual on Screen:**
  - *[2:25 – 2:45]* Show the **Model Performance Matrix** slide (`sih_performance_matrix_slide.png`) and latency benchmarks (5.2 ms inference, 3,451 complaints/min).
  - *[2:45 – 3:00]* Return to closing title screen with Team Name, Problem Statement ID SIH26184, and GitHub repo link.
* **Voiceover:**
  > "MuleShield AI is built for immediate real-world deployment. It runs on commodity CPU hardware with an end-to-end inference latency of just 5.2 milliseconds and an ingestion throughput of over 3,400 complaints per minute — 621 times faster than peak national intake.
  > 
  > Crucially, it consumes the exact data fields already captured by the NCRP, requiring zero new citizen burden.
  > 
  > MuleShield AI transforms reactive complaint logging into proactive, anticipatory law enforcement — intercepting stolen funds before the cash leaves the machine.
  > 
  > Thank you."

---

## 🎙️ Teleprompter-Ready Script (Continuous Text)

> "Every single day across India, thousands of citizens fall victim to cyber-financial fraud. When a victim dials the national 1930 helpline, money has already been laundered through multiple mule accounts within minutes.
> 
> Current systems like CFCFRMS can freeze accounts if money is still in the bank, and tools like Pratibimb only show where crimes happened in the past. But once a syndicate runner physically withdraws paper currency from an ATM, the digital trail goes cold.
> 
> There is an operational void: No existing system tells police which physical ATM to intercept, and when.
> 
> To solve this, we built **MuleShield AI** — the first predictive analytics framework that forecasts physical cash withdrawal locations in advance, enabling proactive law enforcement interdiction during the golden hour.
> 
> MuleShield AI integrates four breakthrough innovations: Deterministic Multi-Hop Graph Traversal, Inductive GraphSAGE Embeddings achieving an F1 of 0.905, Nested Logit Discrete Choice Modeling providing 87.3% physical search zone containment, and a Dual-Regime Survival Countdown offering a median 40 minutes of operational warning lead time.
> 
> Let's see MuleShield AI in action.
> 
> Here is our live operations console. As a complaint is ingested, our deterministic BFS engine immediately traces the multi-hop fund dispersal across banking boundaries. 
> 
> In this transaction graph, our GraphSAGE model instantly pinpoints the terminal cash-out mule holding the remaining funds.
> 
> Moving to the spatial intelligence view, MuleShield AI narrows 1,000 candidate terminals down to just 5 high-probability ATMs on an interactive GIS map. Alongside it, our survival model starts a live tactical countdown — here giving police 34 minutes of advance warning.
> 
> With a single click, the duty officer dispatches ground patrol units to the top search zone. Simultaneously, our switch integration architecture proposes an ISO 8583 Field 39 deceptive response code — such as 'Do Not Honor' — suppressing the cash dispenser and buying crucial minutes for field interception.
> 
> Every action is immutably anchored in a SHA-256 ledger, generating a legally admissible certificate under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023.
> 
> MuleShield AI is built for immediate real-world deployment. It runs on commodity CPU hardware with an end-to-end inference latency of just 5.2 milliseconds and an ingestion throughput of over 3,400 complaints per minute — 621 times faster than peak national intake. Crucially, it consumes the exact data fields already captured by the NCRP, requiring zero new citizen burden.
> 
> MuleShield AI transforms reactive complaint logging into proactive, anticipatory law enforcement — intercepting stolen funds before the cash leaves the machine.
> 
> Thank you."

---

## 💡 Quick Tips for YouTube Video Recording
1. **Screen Resolution:** Record your browser in 1080p (1920×1080) at 60 fps (OBS Studio or Windows Game Bar `Win + G`).
2. **Audio:** Use a clean microphone or headset. Speak with confident, measured pacing.
3. **Pointers/Clicks:** Enable cursor highlighting so judges can clearly follow your clicks in the dashboard.
4. **YouTube Settings for SIH:**
   - **Visibility:** Set to **Unlisted** (or Public). *Do NOT set to Private!*
   - **Title:** `SIH26184 | MuleShield AI - Predictive ATM Cash-Out Interception Framework`
   - **Description:** Include the Problem Statement ID, Team Name, and GitHub link: `https://github.com/Muleshieldsih/Muleshield`.
