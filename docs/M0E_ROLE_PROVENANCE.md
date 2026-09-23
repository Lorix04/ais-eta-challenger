# M0E static vessel-role provenance

This note documents **static population-scope checks only**. These sources are not used as arrival ground truth and are never backfilled into time-varying prediction features.

## Primary rule

When an AIS Message-5 `ship type` code is available, M0E follows the ITU/USCG code family used by the decoded feed. Relevant categories include fishing `30`, sailing `36`, pleasure craft `37`, tug `52`, passenger `6x`, cargo `7x`, and tanker `8x`.

Source: USCG Navigation Center, *AIS Class A Ship Static and Voyage Related Data (Message 5)*.

## External checks used only when AIS static role was missing or insufficient

| Vessel | M0E role | Evidence used | Confidence use |
|---|---|---|---|
| BELLE L'ADRIATIQUE | PASSENGER | CroisiEurope's official coastal/maritime fleet page describes MS La Belle de l'Adriatique as a sea-going cruise vessel carrying passengers. | A |
| BRITOIL CONFIDENCE | OFFSHORE_SUPPLY | Britoil's official vessel specification identifies BRITOIL CONFIDENCE as a DP2 AHTS / Offshore Support Vessel. | A |
| URBANO MONTI | SURVEY_OFFSHORE | Orange Marine's official fleet page identifies Urbano Monti as a survey vessel dedicated to submarine-cable route surveys. | A |

Other externally checked small/recreational/fishing roles are recorded in `config/catania_m0e_vessel_roles.csv`. They do not enter the primary ETA cohort, and their role is not used to decide whether the research-gate crossing itself occurred.

## Port-service craft scope

The Catania/Augusta Port System Authority lists pilotage, towage, mooring and boat services as separate technical-nautical services. M0E therefore keeps validated local port-tug calls in the broader Port Intelligence truth set while excluding them from the primary voyage ETA cohort. This is a benchmark-population choice, not a claim that tug calls are invalid port calls.

## Reproducibility note

The exact URLs and access date are recorded in the M0E report/evidence bundle. If this project is delivered externally, pin or archive the static-role evidence where licensing permits; do not assume a live web profile is immutable.
