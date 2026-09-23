# M15 — Technical Defense Playbook (IT)

> **Uso interno post-freeze.** Questo documento prepara il colloquio tecnico e non modifica, sostituisce o riapre la submission M14. Il file finale M14 resta immutabile.

## 1. Tesi da difendere in una frase

Ho separato due problemi che inizialmente sembravano uguali: **(a)** il target aziendale finale, cioè prevedere la `Tracks.eta` fornita come reference label, e **(b)** un ramo di ricerca fisico sulla reale entrata in area porto ricostruita da AIS. Sul target aziendale ho costruito un benchmark causale, target-free nello split, con final test aperto una sola volta; dopo il final non ho più modificato modello, split o prediction.

## 2. Apertura orale — 30 secondi

«Il punto più importante del progetto non è il singolo numero di MAE ma il protocollo. Ho trasformato il dataset in un problema supervisionato causale: una riga per MMSI, target `Tracks.eta`, feature disponibili solo fino a `Tracks.last_update`, split deterministico basato soltanto sull'MMSI e selezione del modello sulla calibration. Il final test di 53 MMSI è stato aperto una sola volta. Il CatBoost current-snapshot selezionato ottiene 312,0 ore di MAE e 25,3 ore di MedAE: la differenza enorme tra le due metriche è spiegata da reference ETA stale/past/far-future, non da un tentativo di nascondere gli outlier. Dopo il final ho fatto solo diagnostica, explainability e freeze riproducibile.»

## 3. Apertura orale — 90 secondi

«All'inizio ho analizzato il feed AIS come problema di ETA fisica di ingresso porto e ho costruito un ramo rigoroso con event reconstruction, Catania e Augusta, split temporali, route-kNN, uncertainty e holdout congelato. Dopo il chiarimento dell'azienda ho però separato quel lavoro dal task primario: l'esercizio richiedeva `Tracks.eta` come ground-truth/reference value.

Per il task primario ho quindi creato 439 esempi supervisionati, uno per MMSI. Le feature storiche sono causali: nessuna `Positions.recorded_at` successiva a `Tracks.last_update`. Lo split 70/15/15 è deterministico tramite hash dell'MMSI, quindi non usa il target. Ho confrontato baseline semplici, Ridge e CatBoost sulla calibration; il CatBoost current-snapshot è stato selezionato prima di aprire il final. Sul final di 53 MMSI: MAE 312,0 h, MedAE 25,3 h e P90 560,4 h.

La heavy tail è reale nel benchmark: 64/439 reference sono già nel passato, 56 sono oltre 7 giorni nel futuro e i 5 errori peggiori spiegano il 79,1% dell'errore assoluto finale. Non ho rimosso queste righe dal risultato headline. M11 le analizza post-hoc; M12 spiega il modello congelato; M14 blocca tutto con hash, manifest e clean-room test. Quindi il valore che difendo è soprattutto la correttezza sperimentale e la chiarezza su cosa il dataset permette — e cosa non permette — di concludere.»

## 4. Storyline consigliata — 8 minuti

### 4.1 Problema e semantica del target — 60 s
- Il target primario finale è la `Tracks.eta` fornita dall'azienda come **reference label**.
- Non viene chiamata ATA osservata: AIS ETA è un'informazione di viaggio stimata.
- Il formato AIS Message 5 codifica ETA come `MMDDHHMM UTC`, senza anno: l'anno deve essere ricostruito rispetto al timestamp di riferimento.
- Il ramo fisico M0–M9 resta separato e secondario.

### 4.2 Dataset supervisionato causale — 60 s
- 439 reference ETA parseable, 439 MMSI, una riga supervisionata per MMSI.
- `Tracks.last_update` = prediction timestamp del benchmark.
- Feature da `Positions` solo se `recorded_at <= last_update`.
- Nessun uso di `Tracks.eta` come feature.
- Split deterministico MMSI-only 321 / 65 / 53.

### 4.3 Model selection — 60 s
- Baseline semplici prima del boosting.
- Selezione esclusivamente sulla calibration.
- Modello strict scelto: CatBoost current-snapshot, 47 alberi, 14 feature.
- Nessun tuning dopo apertura del final.

### 4.4 Risultato finale — 60 s
- n = 53.
- MAE = 312,0 h.
- MedAE = 25,3 h.
- P90 absolute error = 560,4 h.
- La mediana molto più bassa del MAE indica forte heavy tail.
- Non sostituire mai il risultato strict con il subset 0–7 giorni.

### 4.5 Forensics del target — 60 s
- 64/439 (14,6%) reference già nel passato.
- 56/439 >7 giorni nel futuro.
- 19 righe con |orizzonte| >90 giorni.
- 86,1% ETA al minuto `:00`; 93,8% a `:00/:30`; 96,1% su griglia 5-min.
- Nel final i top-5 errori = 79,1% dell'errore assoluto totale.
- Same frozen model sul subset post-hoc 0–7d: 28,6 h MAE, solo diagnostico.

### 4.6 Explainability — 60 s
- `destination_norm` è la feature singola più importante sia per native importance sia per mean |SHAP| calibration.
- A livello gruppo, motion, voyage state e location sono dominanti.
- SHAP spiega il comportamento del modello, **non** la causalità marittima.
- Il causal-history CatBoost pre-final non migliorava il current-snapshot CatBoost sulla calibration strict.

### 4.7 Reproducibility — 60 s
- M14 conserva hash di reference dataset, modello, prediction ledger e freeze M6.
- ZIP finale deterministico con manifest SHA-256 per-file.
- Full repository: 96/96 test M14.
- Clean-room package: 73/73 test M14.
- Raw company CSV esclusi dal bundle: non viene dichiarato un raw-data replay che non è stato realmente eseguito.

### 4.8 Conclusione — 60 s
- Risultato scientificamente difendibile: benchmark strict onesto + diagnostica della qualità della reference.
- Non stabilito: superiorità sul modello interno aziendale, equivalenza con ATA reale, generalizzazione unseen-port, berth/all-fast o safety-critical use.
- Miglior prossimo esperimento: target operativo osservato + maggiore storico + informazioni portuali/PCS/pilotage se disponibili.

## 5. La domanda più difficile: «312 ore di MAE non è un risultato pessimo?»

Risposta consigliata:

«È un risultato scarso se interpretato come errore tipico di navigazione, ma sarebbe fuorviante fermarsi al MAE. Il benchmark strict include tutte le `Tracks.eta` parseable richieste, comprese reference già nel passato o estremamente lontane. La MedAE è 25,3 h e i soli 5 errori peggiori spiegano il 79,1% dell'errore assoluto totale. Per correttezza non ho eliminato queste reference dal numero headline dopo aver visto il final. Ho separato quindi performance strict e diagnostica di qualità del target. Se l'azienda considera quelle reference valide business-label, il 312 h resta il numero corretto da riportare; se invece sono stale/placeholder, serve una policy target concordata prima di addestrare un nuovo benchmark.»

## 6. «Perché CatBoost?»

«Il dataset supervisionato primario è tabellare e piccolo: 321 righe train, non centinaia di migliaia di esempi indipendenti. CatBoost gestisce bene mix numerico/categorico e non richiede una pipeline molto complessa. Soprattutto, non l'ho scelto per fama: l'ho confrontato sulla calibration con baseline più semplici. Il current-snapshot CatBoost ha vinto la selezione strict pre-final, mentre nel diagnostic future-reference una baseline destination-median è risultata competitiva/migliore. Questo è un segnale contro l'overengineering.»

## 7. «Perché non LSTM/Transformer/GNN?»

«Per il task primario finale l'effettivo sample supervisionato è 439 MMSI, con 321 train. Un modello sequence ad alta capacità avrebbe un rapporto rischio/beneficio sfavorevole e avrebbe aumentato il tuning budget. Inoltre il confronto pre-final fra current snapshot e causal history non ha mostrato un vantaggio della storia nel benchmark strict. Quindi aggiungere capacità dopo il final sarebbe metodologicamente sbagliato.»

## 8. «Come hai evitato leakage?»

Rispondere in quattro punti:
1. `Tracks.eta` è target-only.
2. La storia `Positions` è filtrata a `recorded_at <= Tracks.last_update`.
3. Lo split è MMSI-hash-only, quindi non usa ETA, timestamp target o performance.
4. Il final viene aperto una volta; M11–M15 non fanno model selection.

Se chiedono del ramo fisico M0–M9, aggiungere: route templates e trasformazioni supervisionate venivano costruite train-only nei fold temporali.

## 9. «Perché hash split invece di temporal split nel task M10?»

«Il task finale dell'azienda è uno snapshot cross-sectional con una reference per MMSI, non una sequenza di port call con deployment time chiaramente definito. Ho quindi usato un hash deterministico dell'identità MMSI per avere split riproducibile, target-free e stabile. Nel ramo fisico precedente, dove la dimensione temporale era centrale, ho invece usato expanding temporal evaluation. Se il deployment reale richiede future-time generalization, rifarei M10 con un target/versione dati che supporti esplicitamente quel protocollo.»

## 10. «Perché una riga per MMSI?»

«Perché `Tracks` rappresenta lo stato più recente per nave nel dataset del task e la reference è associata a quello stato. Duplicare le molte `Positions` minute-level come esempi indipendenti avrebbe gonfiato artificialmente N e avrebbe replicato la stessa label attraverso righe correlate.»

## 11. «Cosa significa che la ETA AIS non ha anno?»

«Nel Message 5 l'ETA è codificata come mese-giorno-ora-minuto UTC, senza anno. Quindi una stringa tipo `01010000` non determina da sola l'anno. Nel benchmark l'anno è assegnato deterministicamente rispetto al prediction timestamp; M11 misura anche la sensibilità di questa ricostruzione. Questo è uno dei motivi per cui tratto `Tracks.eta` come reference fornita e non come ATA osservata.»

## 12. «SHAP dimostra che destination causa la ETA?»

«No. SHAP attribuisce la prediction del modello alle feature rispetto a un valore atteso. `destination_norm` è importante nel modello congelato, ma questa è model attribution, non causal inference. È coerente con il dominio che la destinazione sia informativa, ma non uso SHAP per dimostrare causalità.»

## 13. «Perché il history model non ha vinto?»

«Nel candidato causal-history pre-final, la strict calibration MAE era peggiore di 6,38 h rispetto al current-snapshot CatBoost. Nel future-reference diagnostic la differenza era solo 0,22 h. Questo suggerisce che, con questo dataset e questa reference, la storia aggiunge poco segnale utile oltre a stato corrente/destinazione. Non significa che la storia non serva in un vero ETA-to-arrival con mesi di AIS.»

## 14. «Come interpreti le prediction negative?»

«Il modello strict è stato allenato su reference che includono ETA nel passato: 64/439 righe. Quindi predire TTE negativo è coerente con il target strict, anche se non sarebbe la semantica desiderata per un ETA operativo futuro. La range finale delle prediction è circa -109,5 h a +93,8 h, mentre le reference finali vanno circa da -3055 h a +3553 h: questa mismatch di supporto spiega una parte della heavy tail.»

## 15. «Hai battuto il modello aziendale?»

«Non posso dirlo. Non ho accesso agli input, al protocollo di valutazione o alle prediction del modello interno. Posso difendere il mio benchmark, non una superiority claim non misurata.»

## 16. Ramo fisico M0–M9: come raccontarlo senza confondere il task

Usarlo solo come prova di profondità metodologica:
- ricostruzione event-level di arrivi Catania/Augusta;
- 83 call ETA-eligible, 58 unique vessels;
- route-kNN train-only aveva segnale in development;
- residual ML M3 peggiorava e fu scartato;
- holdout M6: 36,5 min route vs 37,7 min geodesic su 13 scorable call;
- gain complessivo non statisticamente risolto;
- nessuna pretesa di equivalenza col target M10.

Messaggio: «Quando il target è fisico, preferisco definire prima l'evento e la leakage boundary; quando l'azienda ha chiarito che la label era `Tracks.eta`, ho versionato il problema invece di riscrivere retroattivamente il ramo precedente.»

## 17. Whiteboard — schema da disegnare in 45 secondi

```text
Positions history (t <= last_update) -----> causal features ----+
                                                               |
Tracks current state ---------------------> current features ---+--> model --> predicted TTE
                                                               |
Tracks.eta -------------------------------> TARGET ONLY --------+

MMSI hash --> train 321 | calibration 65 | final 53
                         select here        open once
```

A fianco scrivere:

```text
STRICT FINAL: MAE 312.0 h | MedAE 25.3 h | P90 560.4 h
M11: top-5 errors = 79.1% total AE
M12: destination_norm top single feature
M14: immutable SHA-256 freeze
```

## 18. Cose da NON dire

- «312 ore è colpa dei dati.» → dire: *il benchmark contiene una heavy tail di reference con semantica problematica; il risultato strict resta valido come benchmark della label fornita.*
- «Gli outlier sono sbagliati.» → non lo sappiamo; usare *stale/past/far-future / placeholder-like diagnostics*.
- «SHAP dimostra la causalità.»
- «Ho battuto il modello aziendale.»
- «Il subset 0–7d è il vero risultato.»
- «L'ETA AIS è ATA.»
- «Il progetto generalizza a tutti i porti.»
- «Il modello è pronto per safety-critical navigation.»

## 19. Domande intelligenti da fare all'intervistatore

1. In produzione, la reference che volete prevedere è ancora `Tracks.eta` oppure avete ATA/port-entry osservata?
2. Qual è l'orizzonte operativo che conta: <6 h, 6–24 h, 1–7 d o tutti?
3. Quanto pesa una prediction molto errata rispetto a stabilità/revision rate?
4. Avete storico completo delle modifiche ETA/destination Message 5?
5. Il modello interno usa dati che il take-home non include: PCS, berth schedule, pilotage, meteo, congestion?
6. La valutazione è per row, vessel, voyage o port call?
7. Come trattate ETA già scadute o placeholder-like?
8. Preferite un modello che rifiuti/flagghi input incoerenti invece di forzare sempre una ETA?

## 20. Fact-check esterno da ricordare

- USCG NAVCEN, AIS Message 5: ETA è `MMDDHHMM UTC`, destination è voyage-related; l'anno non è incluso.
  https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- IMO MSC.74(69): destination e ETA sono voyage-related; le informazioni voyage-related sono aggiornate periodicamente/quando modificate.
  https://wwwcdn.imo.org/localresources/en/OurWork/Safety/Documents/AIS/Resolution%20MSC.74%2869%29.pdf
- CatBoost: SHAP values sono contributi delle feature alla prediction e sommano, insieme all'expected value, alla prediction dell'oggetto.
  https://catboost.ai/docs/en/concepts/shap-values
- Reproducible Builds: output byte-identical possono essere verificati tramite checksum crittografici.
  https://reproducible-builds.org/docs/checksums/
