# M16 — Nota challenger per l'azienda

## Messaggio breve

Dopo aver congelato la soluzione ufficiale M14, ho sviluppato un **challenger separato** per verificare se la struttura del problema AIS potesse essere sfruttata meglio di un singolo modello tabellare. Non ho riutilizzato i 53 casi del final M10 per scegliere il nuovo sistema: tutta la selezione M16 avviene sui soli **386 casi development** con valutazione out-of-fold/nested.

Il challenger combina quattro segnali differenti: **destinazione canonica e prior gerarchici**, **similarità con rotte AIS storiche**, **stima fisica distanza/velocità quando applicabile** e un **modello tabellare robusto**. Un Mixture of Experts sceglie/combina questi segnali usando solo prediction OOF. Sopra il point estimate ho aggiunto un intervallo P10/P50/P90 e un livello di confidence.

## Risultato da citare correttamente

Sul benchmark development-only congelato, il Mixture of Experts passa da **192.93 h MAE** del miglior singolo expert a **185.88 h MAE**, con un miglioramento di **7.04 h** e vittoria in **4/5 fold**. Rispetto al semplice destination-median iniziale (202.97 h), il guadagno è **17.09 h**. L'intervallo centrale nominale all'80% raggiunge **80.05%** di coverage pooled.

## La parte importante: cosa non sto sostenendo

Non presenterei questo numero come prova definitiva che M16 sia migliore su dati futuri. Il red-team mostra che il guadagno sopravvive a leave-one-fold-out, trimming e winsorisation, ma il bootstrap paired al 95% attraversa ancora zero (**[-2.02, 20.16] h**) e una parte del miglioramento è concentrata nella coda.

Per questo la conclusione corretta è: **M16 è un challenger tecnicamente promettente, leakage-safe e più strutturato; la superiorità va confermata su un nuovo holdout mai visto.**

## Perché considero l'approccio più intelligente

Non ho semplicemente aumentato gli hyperparameter o sostituito CatBoost con un algoritmo più complesso. Ho separato il problema in expert con bias differenti e interpretabili:

- normalizzazione semantica delle destinazioni AIS;
- prior robusti con shrinkage per categorie rare;
- matching di rotte storiche reali tramite trajectory similarity;
- expert fisico usato solo quando destination e moto sono sufficientemente affidabili;
- challenger tabellare;
- gating/stacking addestrato esclusivamente su prediction OOF;
- incertezza e confidence esplicite.

## Proposta di validazione

Se volete valutare il challenger, la verifica più pulita è fornire **nuovi dati o un nuovo holdout**. Io manterrei M16 congelato, fisserei prima metrica e target, e farei un'unica valutazione senza ulteriore tuning.

**M14 resta la submission ufficiale congelata. M16 è un challenger aggiuntivo, non una riscrittura post-hoc del risultato ufficiale.**
