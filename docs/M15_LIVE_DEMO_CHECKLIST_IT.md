# M15 — Live Demo & Technical Defense Checklist (IT)

> **Obiettivo:** mostrare rigore e riproducibilità senza accidentalmente modificare il freeze M14 o richiedere i CSV aziendali assenti.

## Prima della call

- [ ] Aprire `EXECUTIVE_SUMMARY.md` e `TAKE_HOME_REPORT.md`.
- [ ] Tenere aperto `reports/M10_REPORT.md` per il benchmark primario.
- [ ] Tenere aperto `reports/M11_REPORT.md` per heavy-tail/reference quality.
- [ ] Tenere aperto `reports/M12_REPORT.md` per explainability.
- [ ] Tenere aperto `reports/M14_REPRODUCIBILITY_REPORT.md` per freeze/reproducibility.
- [ ] Tenere questo playbook e `M15_QA_BANK_IT.md` fuori dalla condivisione schermo se non necessario: sono note interne.

## Demo sicura — 3 minuti

### 1. Verifica freeze

```bash
python scripts/67_m15_verify_defense.py
```

Expected: tutti gli hash M14/M10/M6 invariati e i documenti M15 coerenti.

### 2. Test repository

```bash
PYTHONPATH=src python -m pytest -q
```

Expected dopo M15: suite completa PASS.

### 3. Verifica submission M14 senza rebuild

```bash
python scripts/65_m14_verify_final_submission.py
```

Questo verifica il pacchetto M14 già congelato; non usarlo per introdurre nuove modifiche.

## Comandi da NON usare durante la demo

- `scripts/50_m10_verify.py` se i raw CSV non sono presenti: richiede `data/vessel_tracks.csv`.
- Qualunque script di training/benchmark che possa riscrivere model artifacts.
- `scripts/64_m14_build_final_submission.py` durante la call: il file M14 è già congelato; non serve rebuildarlo per dimostrare integrità.

## Percorso visuale consigliato

1. `reports/m10_result_panel.png` — risultato primario.
2. `reports/m11_final_error_concentration.png` — mostra heavy tail.
3. `reports/m12_feature_importance.png` — attribution del frozen model.
4. `reports/m14_result_panel.png` — freeze/reproducibility.
5. `reports/m15_defense_map.png` — mappa narrativa del colloquio.

## Se chiedono codice

Mostrare in questo ordine:
1. costruzione dataset causale M10;
2. split target-free;
3. model selection calibration-only;
4. freeze/open marker;
5. M12 prediction reproduction;
6. M14 package verifier.

Non iniziare dal CatBoost: iniziare dal **data contract** e dal **leakage boundary**.

## Se chiedono «fammi vedere dove impedisci il leakage»

Mostrare il filtro temporale nel builder M10 e spiegare verbalmente:

```text
Positions.recorded_at <= Tracks.last_update
```

Poi mostrare che `Tracks.eta` non è una feature e che lo split usa solo l'hash MMSI.

## Se chiedono «fammi vedere che non hai ritoccato il final»

Mostrare:
- `reports/M10_FREEZE.json`;
- `reports/M14_FINAL_FREEZE.json`;
- SHA-256 del modello `efbb...f86a`;
- SHA-256 ledger final predictions `fc63...ee1`;
- SHA-256 ZIP M14 `6568...553c`.

## Se chiedono «dove sono i raw data?»

«Non sono nel submission bundle per confidentiality/size. Il progetto distingue data-free verification da raw-data replay. Per questo il clean-room package può verificare codice, manifest e analytical artifacts, ma non dichiaro una rigenerazione end-to-end dai tre CSV aziendali quando quei CSV non sono presenti.»

## Checklist post-call

- [ ] Non modificare il frozen M14 in-place.
- [ ] Se arrivano nuove definizioni target/dati, creare nuova milestone/versione.
- [ ] Non usare il final M10 come calibration del nuovo esperimento.
- [ ] Conservare le nuove assunzioni come machine-readable state/evidence.
