# M15 — Interview / Technical Defense Q&A Bank (IT)

> **Uso interno.** Ordine: Tier A = quasi certo / critico; Tier B = approfondimento ML/data; Tier C = maritime/domain/reproducibility.

## Tier A — Le 20 domande da saper rispondere senza esitazione

### A01 — Qual era il task finale?
**Risposta orale:** «Prevedere la `Tracks.eta` fornita dall'azienda come reference label, usando lo stato corrente in `Tracks` e solo la storia `Positions` disponibile fino a `Tracks.last_update`. Non ho trattato quella ETA come ATA osservata.»

### A02 — Quanti esempi avevi?
«439 reference parseable su 439 MMSI, una riga supervisionata per MMSI. Split: 321 train, 65 calibration, 53 final.»

### A03 — Perché non dici di avere ~1,5 milioni di training examples?
«Perché quelle sono osservazioni AIS fortemente correlate. Il task M10 ha una reference per MMSI; replicarla su ogni posizione avrebbe creato pseudo-replication e leakage-like weighting.»

### A04 — Qual è il risultato headline?
«Strict final su 53 MMSI: MAE 312,0 h, MedAE 25,3 h, P90 absolute error 560,4 h.»

### A05 — Perché MAE e MedAE sono così distanti?
«Heavy tail del target/reference: 64/439 ETA già nel passato, 56 >7 giorni nel futuro; sul final i top-5 errori spiegano il 79,1% dell'errore assoluto.»

### A06 — Perché non hai tolto gli outlier?
«Perché il task strict richiedeva tutte le reference parseable. Togliere righe dopo aver visto il final avrebbe cambiato il benchmark post-hoc. Le ho invece analizzate in M11.»

### A07 — Che modello hai scelto?
«CatBoost current-snapshot, 47 alberi e 14 feature, selezionato su calibration prima di aprire il final.»

### A08 — Quali feature contano di più?
«Come singola feature `destination_norm`; poi location/motion/nav status. A gruppi, sulla calibration, motion è il gruppo con mean |SHAP| più alto. Ma sono attribution, non causalità.»

### A09 — Come hai evitato leakage?
«Target-only ETA, cutoff temporale su Positions, MMSI-hash split target-free, model selection solo su calibration, final aperto una volta.»

### A10 — Perché hash split?
«Per avere partizione stabile, riproducibile e indipendente dal target nel task cross-sectional one-row-per-MMSI. Il ramo physical-arrival usa invece split temporali quando il deployment temporale è il problema.»

### A11 — Perché current snapshot invece di history?
«Il candidato causal-history pre-final non migliorava la strict calibration: +6,38 h MAE rispetto al current snapshot. Non c'era evidenza per aumentare complessità.»

### A12 — Perché non deep learning?
«321 righe train effettive nel benchmark primario e nessun gain della storia pre-final: più capacità avrebbe aumentato rischio di overfit e tuning senza evidenza.»

### A13 — Cosa significa `Tracks.eta` tecnicamente?
«È la reference ETA dell'esercizio. AIS Message 5 rappresenta ETA come mese/giorno/ora/minuto UTC, senza anno, quindi non è un timestamp osservato di arrivo.»

### A14 — Hai confrontato col modello aziendale?
«No. Non avevo le sue prediction, input o protocollo di scoring; quindi non faccio claims di superiorità.»

### A15 — Cos'è il subset 0–7 giorni?
«Una diagnostica post-hoc M11 sullo stesso modello congelato: 40 final rows, MAE 28,6 h. Non sostituisce il risultato strict.»

### A16 — Perché prediction negative?
«Il target strict contiene ETA già nel passato. CatBoost impara anche TTE negative. In un prodotto operational future-ETA metterei una target policy diversa, ma solo prima di un nuovo freeze.»

### A17 — Cosa hai fatto dopo il final?
«Solo forensics M11, explainability M12, reconciliation M13, reproducibility freeze M14 e interview prep M15. Nessun training/tuning/reselection.»

### A18 — Come garantisci riproducibilità?
«Hash dei frozen artifacts, ZIP deterministico, manifest SHA-256 per file, verifier nel package, test in clean-room.»

### A19 — Cosa non è riproducibile senza i CSV originali?
«Il raw-data replay end-to-end. Il bundle li esclude per confidentiality/size, quindi non dichiaro un replay che non posso eseguire.»

### A20 — Cosa faresti con più tempo/dati?
«Definirei un target operativo osservato, raccoglierei mesi di AIS + storico ETA Message 5 + port-call/PCS/pilotage/berth data, poi rifarei split e benchmark prima di aumentare la capacità del modello.»

## Tier B — ML / statistics / engineering

### B01 — Perché MAE come selection metric?
«È interpretabile nella stessa unità dell'ETA e meno dominata dagli estremi di RMSE. Ho comunque riportato MedAE, P90 e threshold accuracies per mostrare la distribuzione.»

### B02 — Perché non ottimizzare MedAE se è più stabile?
«Avrebbe cambiato l'obiettivo. La selection metric era fissata pre-final come calibration MAE; dopo aver visto la heavy tail non l'ho cambiata.»

### B03 — Il bootstrap selected-vs-destination baseline cosa dice?
«Sul strict final il mean gain stimato del CatBoost vs destination median è ~41,7 h, ma CI95 va circa da -8,2 a 126,8 h: non lo tratto come superiorità statisticamente risolta.»

### B04 — Perché CatBoost categorical invece di one-hot + XGBoost?
«CatBoost riduce engineering manuale per categoriche e gestisce bene small tabular data. Ma la scelta è empirica: il punto non è il brand del modello, è il protocollo calibration-before-final.»

### B05 — Come hai interpretato SHAP?
«Per ogni oggetto SHAP decompone la prediction in expected value + contributi feature. Ho verificato additivity numericamente. Non lo uso per causal inference.»

### B06 — Qual è l'expected value SHAP?
«Circa 40,41 h nel frozen model.»

### B07 — Qual è il range prediction vs target nel final?
«Prediction circa -109,5 a +93,8 h; reference circa -3055 a +3553 h. Il modello non può inseguire bene quegli estremi con il supporto train disponibile.»

### B08 — Perché non clipparli?
«Un clipping post-hoc scelto dopo aver visto il final sarebbe tuning. Un eventuale clipping deve essere parte di una nuova policy definita e validata su development/calibration.»

### B09 — Perché una calibration separata?
«Per separare model selection/tuning dal final evaluation. Train addestra, calibration decide, final misura una sola volta.»

### B10 — Perché non cross-validation sul primario?
«Si poteva usare per development, ma con il protocollo scelto ho mantenuto una calibration esplicita e un final untouched. La priorità era avere una boundary chiara e auditabile.»

### B11 — Il MMSI può essere una leakage feature?
«Non viene usato come predittore; viene usato solo per split deterministico. Questo evita che identità nave entri direttamente nel modello.»

### B12 — `destination_norm` rischia target leakage?
«No se è disponibile al prediction time ed è distinta dalla ETA. È molto informativa ma non contiene il valore target. È comunque soggetta a qualità/manual-entry del dato.»

### B13 — Come gestisci missing/categorical?
«La pipeline M10 normalizza categoriche e passa feature coerenti al CatBoost. La validazione automatica controlla schema e invarianti del dataset derivato.»

### B14 — Perché non usare il target ETA storico da Positions?
«L'azienda ha chiarito che lo storico `Positions` contiene destination ma non historical ETA. Retro-propagare la ETA di `Tracks` alle posizioni storiche sarebbe future leakage.»

### B15 — Cosa indica l'exact-hour fraction 86,1%?
«Che la reference ha forte quantizzazione/manual-entry pattern. È un segnale di qualità/granularità della label, non una prova automatica che sia errata.»

### B16 — Perché il top-5 share è utile?
«Mostra che il MAE aggregato è fortemente concentrato in pochissimi casi. Aiuta a distinguere errore tipico da heavy-tail failure.»

### B17 — Perché non winsorize?
«Stesso motivo del clipping: cambierebbe la funzione di valutazione strict. Va definito prima, non dopo il final.»

### B18 — Cosa significa frozen model?
«Hash del file `.cbm` invariato da M10 a M15. M12 ricalcola attribution/prediction senza refit.»

### B19 — Come verifichi che M15 non abbia cambiato il modello?
«Confronto SHA-256 di modello, reference dataset, prediction ledger, M6 freeze e dell'intero ZIP M14 prima/dopo M15.»

### B20 — Cosa sarebbe un nuovo esperimento valido?
«Nuova target version, nuovo split/freeze, criteri pre-specificati e nuovo final test. Non riutilizzerei il final M10 per scegliere modifiche.»

## Tier C — Maritime / AIS / physical branch

### C01 — Cos'è AIS Message 5?
«Messaggio Class A static/voyage-related: include, tra gli altri, draught, destination e ETA. L'ETA è MMDDHHMM UTC.»

### C02 — Perché manca l'anno?
«Lo standard codifica mese, giorno, ora e minuto; il receiver/application deve collocarlo nel contesto temporale. Da qui la missing-year sensitivity.»

### C03 — Quanto spesso arriva voyage-related data?
«Le linee IMO descrivono aggiornamento periodico e quando i dati vengono modificati; il feed provider può però avere semantics di polling/snapshot proprie.»

### C04 — Che differenza c'è tra AIS ETA e ATA?
«ETA è una stima dichiarata/di viaggio; ATA è un evento osservato di arrivo. Sono oggetti diversi.»

### C05 — Cosa hai fatto nel ramo physical ETA?
«Ho ricostruito port-call event, geometrie Catania/Augusta, route physics, train-only route-kNN, uncertainty conformal e holdout congelato.»

### C06 — Risultato M6?
«13/16 final calls scorable; route 36,45 min MAE vs geodesic 37,75 min. Il gain ~3,4% non è statisticamente risolto.»

### C07 — Perché 3/16 non scorable?
«La scope causale richiedeva un inbound-approach state valido. Non ho allargato la regola dopo aver visto il holdout.»

### C08 — Perché route-kNN?
«Distanza marittima reale e corridoi storici possono differire dalla geodesica; in development route-kNN mostrò segnale consistente, soprattutto su cold vessels.»

### C09 — Perché residual ML fu scartato?
«Nel second-stage OOF peggiorava il route-physics predictor; il CI del gain era negativo. Quindi M3 concluse che la complessità non era giustificata.»

### C10 — Perché non generalizzi unseen port?
«Le geometrie e route erano port-specifiche; non c'era evidenza di leave-one-port-out generalization e non la dichiaro.»

## Tier D — Reproducibility / software engineering

### D01 — Cosa contiene il final ZIP M14?
«Codice/analytical artifacts necessari alla review, manifest content hash e verifier; esclude raw AIS, high-volume derived data e internal evidence.»

### D02 — Perché checksum esterno del ZIP?
«Un archive hash dentro lo stesso archive creerebbe dipendenza self-referential. Il checksum byte-level è quindi esterno.»

### D03 — Cosa vuol dire deterministic ZIP?
«Ordine member stabile, timestamp ZIP fisso, mode fisso, contenuto stabile. Due build consecutive producono lo stesso SHA-256.»

### D04 — Quanti test passavano a M14?
«96/96 full repository; 73/73 dal package estratto clean-room.»

### D05 — Perché `compileall` oltre a pytest?
«Per intercettare errori sintattici/importabili in file che una suite mirata potrebbe non importare direttamente.»

### D06 — Perché evidence bundle?
«Ogni milestone materiale conserva change record, patch, commands, test output e screenshot, così le decisioni sono auditabili.»

## Risposte da evitare

- «I dati sono sbagliati.»
- «Gli outlier vanno rimossi.»
- «Il CatBoost è migliore in assoluto.»
- «SHAP dice cosa causa l'arrivo.»
- «36 minuti è il risultato finale del take-home.»
- «28,6 h è il risultato finale.»
- «Ho usato 1,5 milioni di esempi.»
- «Il modello è production-ready.»

## Numeri da memorizzare

| Voce | Valore |
|---|---:|
| Reference parseable | 439 |
| Split | 321 / 65 / 53 |
| Strict final MAE | 312,0 h |
| Strict final MedAE | 25,3 h |
| Strict final P90 AE | 560,4 h |
| Past reference | 64 / 439 |
| Future >7d | 56 / 439 |
| Top-5 final AE share | 79,1% |
| 0–7d diagnostic | 40 rows, 28,6 h MAE |
| Frozen CatBoost | 47 trees, 14 features |
| Top single feature | `destination_norm` |
| M14 repo tests | 96/96 |
| M14 clean package tests | 73/73 |
| M14 final ZIP SHA-256 | `65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c` |
