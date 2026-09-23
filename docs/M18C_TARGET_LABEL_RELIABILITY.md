# M18C Target & Label Reliability Policy

M18C is a diagnostic-only audit of the company-provided `Tracks.eta` reference. It does not define a new target.

## Frozen interpretation

- `Tracks.last_update` remains the decision time.
- `Tracks.eta` remains the strict company reference target under M18A.
- AIS Message 5 encodes ETA as MMDDHHMM UTC and does not encode a year; year resolution therefore remains an explicit project assumption.
- An ETA timestamp without an authoritative target location/event must not be silently described as ATA, pilot-boarding arrival, berth arrival or all-fast.

## Diagnostic tiers

M18C assigns reference-only reliability tiers for analysis. They are prohibited as model inputs and prohibited as training weights unless a future milestone explicitly freezes a new protocol before evaluating outcomes.

## AIS research-gate events

The manually validated Catania/Augusta gate events are valuable behavioral evidence, but they are not official port-call records. M18C uses them only to test temporal alignment with company reference timestamps. They do not replace `Tracks.eta` and do not establish an authoritative ATA.

## External basis

- USCG/NAVCEN AIS Message 5: ETA is MMDDHHMM UTC: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- IMO JIT/FAL model: arrival/planned/requested timestamps are tied to specified locations including Pilot Boarding Place and berth: https://imocompendium.imo.org/public/IMO-Compendium/Current/DS/Just%20In%20Time%20Concept/d11.htm
- Chu, Yan & Wang (2025), Transportation Research Part C: vessel-arrival prediction framework fusing AIS and port-call records and analysing vessel-reported ETA accuracy: https://www.sciencedirect.com/science/article/abs/pii/S0968090X25001329
