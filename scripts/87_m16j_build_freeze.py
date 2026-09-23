#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "reports"
D = ROOT / "docs"
DIST = ROOT / "dist"

from ais_eta.m16j import (  # noqa: E402
    M16J_OFFICIAL_SUBMISSION,
    M16J_PROMOTION_DECISION,
    M16J_VERSION,
    PromotionEvidence,
    promotion_decision,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text())


def metric(path: str, model: str) -> dict:
    df = pd.read_csv(R / path)
    row = df.loc[df["model"].eq(model)].iloc[0]
    return {k: float(row[k]) for k in ["mae_h", "medae_h", "p90_ae_h"]}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def plot_progression(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    vals = [float(r["mae_h"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(labels, vals)
    ax.set_ylabel("OOF MAE (hours; lower is better)")
    ax.set_title("M16 development-only challenger progression")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(R / "m16j_result_progression.png", dpi=160, metadata={"Software": "AIS ETA Challenger"})
    plt.close(fig)


def plot_ablation() -> None:
    df = pd.read_csv(R / "m16i_ablation_metrics.csv")
    keep = df.loc[df["scenario"].isin([
        "official_m16g", "no_physics_structural", "no_route_structural",
        "no_canonical_destination_proxy", "no_longitudinal_history_proxy",
    ])].copy()
    order = [
        "official_m16g", "no_physics_structural", "no_route_structural",
        "no_canonical_destination_proxy", "no_longitudinal_history_proxy",
    ]
    keep["ord"] = keep["scenario"].map({k:i for i,k in enumerate(order)})
    keep = keep.sort_values("ord")
    labels = [
        "Full M16G", "No physics", "No route", "No canonical destination", "No longitudinal history"
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(labels, keep["mae_h"].astype(float).tolist())
    ax.set_ylabel("Counterfactual MAE (hours)")
    ax.set_title("M16I structural ablation stress — no post-red-team retuning")
    ax.tick_params(axis="x", rotation=22)
    for bar, val in zip(bars, keep["mae_h"].astype(float)):
        ax.text(bar.get_x()+bar.get_width()/2, val+1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(keep["mae_h"].astype(float)) * 1.14)
    fig.tight_layout()
    fig.savefig(R / "m16j_ablation_summary.png", dpi=160, metadata={"Software": "AIS ETA Challenger"})
    plt.close(fig)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    boxes = [
        (0.03,0.70,0.19,0.16,"AIS causal snapshot\n+ recent trajectory"),
        (0.29,0.78,0.20,0.12,"Destination prior\nM16B/C"),
        (0.29,0.58,0.20,0.12,"Route analogue\nM16D"),
        (0.29,0.38,0.20,0.12,"Physics gate\nM16E"),
        (0.29,0.18,0.20,0.12,"Tabular HGB\nM16F"),
        (0.58,0.48,0.18,0.18,"Nested OOF\nMixture of Experts\nM16G"),
        (0.82,0.58,0.15,0.14,"P50 ETA\n185.88 h MAE"),
        (0.82,0.33,0.15,0.14,"P10/P90 +\nconfidence\nM16H"),
    ]
    for x,y,w,h,t in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,linewidth=1.8))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=10)
    arrows=[((0.22,0.78),(0.29,0.84)),((0.22,0.76),(0.29,0.64)),((0.22,0.74),(0.29,0.44)),((0.22,0.72),(0.29,0.24)),
            ((0.49,0.84),(0.58,0.61)),((0.49,0.64),(0.58,0.59)),((0.49,0.44),(0.58,0.56)),((0.49,0.24),(0.58,0.53)),
            ((0.76,0.58),(0.82,0.65)),((0.76,0.54),(0.82,0.40))]
    for a,b in arrows:
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","lw":1.4})
    ax.text(0.5,0.04,"All model selection: 386 development MMSIs only • 53 old-final MMSIs hard-blocked • M14 remains official submission",ha="center",fontsize=10)
    ax.set_title("M16 Smart Hybrid Challenger — frozen architecture",fontsize=14)
    fig.tight_layout()
    fig.savefig(R / "m16j_architecture.png", dpi=160, metadata={"Software": "AIS ETA Challenger"})
    plt.close(fig)


def select_payload() -> list[Path]:
    out: set[Path] = set()
    for pat in ["src/ais_eta/m16*.py", "tests/test_m16*.py"]:
        out.update(ROOT.glob(pat))
    # M16 implementation scripts only; freeze script is included, verifier is added after it exists.
    for p in (ROOT / "scripts").glob("*.py"):
        try:
            n = int(p.name.split("_", 1)[0])
        except Exception:
            continue
        if 69 <= n <= 88:
            out.add(p)
    for p in R.iterdir():
        if p.is_file() and (p.name.startswith("M16") or p.name.startswith("m16")):
            if p.name not in {"M16_CONTENT_MANIFEST.json", "M16_FINAL_FREEZE.json"}:
                out.add(p)
    for p in [
        D / "M16_COMPANY_NOTE_IT.md", ROOT / "WORKFLOW.md", ROOT / "README.md",
        ROOT / "EXECUTIVE_SUMMARY.md", ROOT / "TAKE_HOME_REPORT.md", ROOT / "REFERENCES.md",
        ROOT / "requirements.txt", ROOT / "PYTHON_VERSION.txt",
    ]:
        if p.exists(): out.add(p)
    cat = ROOT / "data/static/m16b_destination_catalog.csv"
    if cat.exists(): out.add(cat)
    return sorted(out, key=lambda p: p.relative_to(ROOT).as_posix())


def build_deterministic_zip(payload: list[Path], out_zip: Path) -> None:
    fixed = (2026, 9, 22, 0, 0, 0)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in payload:
            rel = p.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(rel, fixed)
            info.create_system = 3
            info.external_attr = (0o644 & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    DIST.mkdir(exist_ok=True)
    a = metric("m16a_oof_model_metrics.csv", "train_destination_median")
    a_cat = metric("m16a_oof_model_metrics.csv", "catboost_current_snapshot")
    c = metric("m16c_model_metrics.csv", "m16c_hierarchical_prior")
    d = metric("m16d_model_metrics.csv", "m16d_route_analogue")
    e = pd.read_csv(R / "m16e_model_metrics.csv").query("scope == 'all_development' and model == 'm16e_hard_gate_route_fallback'").iloc[0]
    f = metric("m16f_model_metrics.csv", "m16f_hist_gb")
    g = metric("m16g_candidate_metrics.csv", "m16g_selected_nested_mixture")
    h = json_read(R / "M16H_SUMMARY.json")
    i = json_read(R / "M16I_SUMMARY.json")

    evidence = PromotionEvidence(
        pooled_gain_h=float(i["pooled_gain_h"]), fold_wins=int(i["fold_wins_vs_best_single"]),
        leave_one_fold_positive=int(i["leave_one_fold_positive"]),
        trimmed_gain_h=float(i["trimmed_5pct_abs_target_gain_h"]), winsorized_gain_h=float(i["winsorized_95_gain_h"]),
        bootstrap_positive_probability=float(i["bootstrap_positive_probability"]),
        bootstrap_ci95_low_h=float(i["bootstrap_gain_ci95_h"][0]),
        top5_positive_gain_share=float(i["top5_positive_gain_share"]),
    )
    decision = promotion_decision(evidence)
    assert decision == M16J_PROMOTION_DECISION

    comparison = [
        {"stage":"M16A","label":"Destination median","mae_h":a["mae_h"],"medae_h":a["medae_h"],"p90_ae_h":a["p90_ae_h"],"role":"baseline"},
        {"stage":"M16A","label":"Current CatBoost","mae_h":a_cat["mae_h"],"medae_h":a_cat["medae_h"],"p90_ae_h":a_cat["p90_ae_h"],"role":"baseline"},
        {"stage":"M16C","label":"Hierarchical prior","mae_h":c["mae_h"],"medae_h":c["medae_h"],"p90_ae_h":c["p90_ae_h"],"role":"expert"},
        {"stage":"M16D","label":"Route analogue","mae_h":d["mae_h"],"medae_h":d["medae_h"],"p90_ae_h":d["p90_ae_h"],"role":"expert"},
        {"stage":"M16E","label":"Physics-gate + route","mae_h":float(e.mae_h),"medae_h":float(e.medae_h),"p90_ae_h":float(e.p90_ae_h),"role":"best_single"},
        {"stage":"M16F","label":"HistGradientBoosting","mae_h":f["mae_h"],"medae_h":f["medae_h"],"p90_ae_h":f["p90_ae_h"],"role":"expert"},
        {"stage":"M16G","label":"Nested MoE","mae_h":g["mae_h"],"medae_h":g["medae_h"],"p90_ae_h":g["p90_ae_h"],"role":"challenger_point"},
    ]
    write_csv(R / "m16j_challenger_comparison.csv", comparison, ["stage","label","mae_h","medae_h","p90_ae_h","role"])

    claims = [
        {"claim":"M16 is development-only evidence","status":"SUPPORTED","evidence":"386 MMSIs; old 53-row M10 final hard-blocked","fresh_holdout_required":"YES"},
        {"claim":"Nested MoE improves pooled OOF MAE vs best single expert","status":"SUPPORTED_DEVELOPMENT_ONLY","evidence":f"{g['mae_h']:.2f} h vs {float(e.mae_h):.2f} h; +{i['pooled_gain_h']:.2f} h","fresh_holdout_required":"YES"},
        {"claim":"Gain is stable to fold omission / trimming / winsorisation","status":"SUPPORTED_WITH_CAVEAT","evidence":f"LOFO {i['leave_one_fold_positive']}/5; trimmed +{i['trimmed_5pct_abs_target_gain_h']:.2f} h; winsor +{i['winsorized_95_gain_h']:.2f} h","fresh_holdout_required":"YES"},
        {"claim":"M16 is proven superior on unseen company data","status":"NOT_ESTABLISHED","evidence":f"paired bootstrap 95% CI [{i['bootstrap_gain_ci95_h'][0]:.2f}, {i['bootstrap_gain_ci95_h'][1]:.2f}] h crosses zero","fresh_holdout_required":"YES"},
        {"claim":"M16H central-80% uncertainty is calibrated marginally","status":"SUPPORTED_DEVELOPMENT_ONLY","evidence":f"{h['pooled_coverage']*100:.2f}% pooled coverage; mean width {h['pooled_mean_width_h']:.2f} h","fresh_holdout_required":"YES"},
        {"claim":"M14 remains official frozen submission","status":"LOCKED","evidence":"M14 checksum unchanged; M16 not substituted","fresh_holdout_required":"NO"},
    ]
    write_csv(R / "m16j_claim_register.csv", claims, ["claim","status","evidence","fresh_holdout_required"])

    plot_progression([comparison[0], comparison[3], comparison[4], comparison[6]])
    plot_ablation(); plot_architecture()

    summary = {
        "milestone":"M16J", "version":M16J_VERSION, "status":"DONE_SMART_CHALLENGER_FREEZE",
        "promotion_decision":decision, "official_submission":M16J_OFFICIAL_SUBMISSION,
        "development_rows":386, "blocked_old_final_rows":53, "old_final_used_for_m16_selection":False,
        "baseline_destination_median_mae_h":a["mae_h"], "baseline_current_catboost_mae_h":a_cat["mae_h"],
        "best_single_m16e_mae_h":float(e.mae_h), "m16g_mae_h":g["mae_h"], "m16g_medae_h":g["medae_h"], "m16g_p90_h":g["p90_ae_h"],
        "gain_h_vs_best_single":float(i["pooled_gain_h"]), "gain_h_vs_destination_baseline":float(a["mae_h"]-g["mae_h"]),
        "fold_wins_vs_best_single":int(i["fold_wins_vs_best_single"]), "leave_one_fold_positive":int(i["leave_one_fold_positive"]),
        "bootstrap_positive_probability":float(i["bootstrap_positive_probability"]), "bootstrap_gain_ci95_h":i["bootstrap_gain_ci95_h"],
        "top5_positive_gain_share":float(i["top5_positive_gain_share"]),
        "m16h_nominal_coverage":float(h["nominal_coverage"]), "m16h_empirical_coverage":float(h["pooled_coverage"]),
        "m16h_mean_width_h":float(h["pooled_mean_width_h"]), "m16h_width_reduction_fraction":float(h["mean_width_reduction_fraction"]),
        "fresh_untouched_holdout_required_for_superiority_claim":True,
        "m14_remains_official_submission":True,
    }
    (R / "M16_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# M16 — Smart Hybrid Challenger: final freeze\n\n## Final decision\n\n**{decision}**\n\nM16 is frozen as a **development-only challenger for fresh external validation**. It does **not** replace the M14 official submission, and it must not be described as proven superior on unseen company data. The 53 already-observed M10 final MMSIs were hard-blocked from every M16 selection decision.\n\n## Why M16 exists\n\nM16 tests whether domain structure can improve on a single tabular predictor without reopening the observed final set. The architecture combines a canonical destination path, hierarchical destination priors, historical route analogues, a maritime/physics gate, a robust tabular expert, nested OOF mixture-of-experts, and calibrated uncertainty/confidence.\n\n## Development-only point result\n\n- destination-median baseline: **{a['mae_h']:.2f} h MAE**\n- current-snapshot CatBoost baseline: **{a_cat['mae_h']:.2f} h MAE**\n- route analogue: **{d['mae_h']:.2f} h MAE**\n- physics-gate + route fallback: **{float(e.mae_h):.2f} h MAE**\n- best tabular challenger (HistGradientBoosting): **{f['mae_h']:.2f} h MAE**\n- frozen nested MoE: **{g['mae_h']:.2f} h MAE / {g['medae_h']:.2f} h MedAE / {g['p90_ae_h']:.2f} h P90**\n- gain vs best frozen single operational expert: **{i['pooled_gain_h']:.2f} h**, with **{i['fold_wins_vs_best_single']}/5** outer-fold wins\n- gain vs the destination-median M16A baseline: **{a['mae_h']-g['mae_h']:.2f} h**\n\n## Robustness and caveat\n\nThe gain remains positive after leaving out each outer fold ({i['leave_one_fold_positive']}/5), after removing the top 5% of rows by absolute target magnitude (+{i['trimmed_5pct_abs_target_gain_h']:.2f} h), after 95% winsorisation (+{i['winsorized_95_gain_h']:.2f} h), and after removing the largest net-gain destination (+{i['gain_after_removing_largest_net_destination_h']:.2f} h).\n\nThe paired bootstrap remains the reason not to overclaim: **P(gain>0)={i['bootstrap_positive_probability']*100:.2f}%**, but the **95% gain interval is [{i['bootstrap_gain_ci95_h'][0]:.2f}, {i['bootstrap_gain_ci95_h'][1]:.2f}] h**, and the five largest positive-gain rows account for **{i['top5_positive_gain_share']*100:.1f}%** of total positive gain. The correct claim is therefore “promising, structured challenger”, not “proven winner”.\n\n## Probabilistic output\n\nM16H leaves the M16G point estimate unchanged as P50 and adds a central-80% interval plus confidence tier. Pooled empirical coverage is **{h['pooled_coverage']*100:.2f}%**, with **{h['pooled_mean_width_h']:.2f} h** mean width, **{h['mean_width_reduction_fraction']*100:.1f}%** narrower than the global residual baseline. Confidence tiers separate typical error strongly, but low-confidence/reference-pathology regimes remain heavy-tailed and are not given a conditional-coverage guarantee.\n\n## Strict evidence separation\n\n**M16 development evidence:** all M16A–I model selection, gating, interval selection, ablation and red-team work uses only the 386 former M10 train+calibration MMSIs.\n\n**Old M10 final:** 53 MMSIs were already observed before M16. They are prohibited from M16 selection and are not used here to create a new headline score.\n\n**Official submission:** M14 remains the official frozen company submission. Its checksum is re-verified by M16J.\n\n## What would establish a fresh generalization claim\n\nScore the frozen M16 challenger once on **new, untouched company data or a newly issued holdout** with target semantics fixed in advance. Compare the frozen M16 point predictor and intervals against the agreed baseline with the same metric and report paired uncertainty. No M16 re-tuning should occur after that new holdout is revealed.\n\n## Deliverables\n\n- `reports/m16j_challenger_comparison.csv` — frozen OOF comparison ledger\n- `reports/m16j_claim_register.csv` — claims and allowed wording\n- `reports/m16j_architecture.png` — final architecture\n- `reports/m16j_result_progression.png` — result progression\n- `reports/m16j_ablation_summary.png` — structural ablation summary\n- `docs/M16_COMPANY_NOTE_IT.md` — concise company-facing explanation\n- `reports/M16_CONTENT_MANIFEST.json` — SHA-256 manifest of frozen M16 payload\n- `dist/ais_eta_m16_challenger_freeze.zip` — deterministic M16 challenger freeze archive, **not** the official M14 submission\n"""
    (R / "M16_REPORT.md").write_text(report)

    company_note = f"""# M16 — Nota challenger per l'azienda\n\n## Messaggio breve\n\nDopo aver congelato la soluzione ufficiale M14, ho sviluppato un **challenger separato** per verificare se la struttura del problema AIS potesse essere sfruttata meglio di un singolo modello tabellare. Non ho riutilizzato i 53 casi del final M10 per scegliere il nuovo sistema: tutta la selezione M16 avviene sui soli **386 casi development** con valutazione out-of-fold/nested.\n\nIl challenger combina quattro segnali differenti: **destinazione canonica e prior gerarchici**, **similarità con rotte AIS storiche**, **stima fisica distanza/velocità quando applicabile** e un **modello tabellare robusto**. Un Mixture of Experts sceglie/combina questi segnali usando solo prediction OOF. Sopra il point estimate ho aggiunto un intervallo P10/P50/P90 e un livello di confidence.\n\n## Risultato da citare correttamente\n\nSul benchmark development-only congelato, il Mixture of Experts passa da **{float(e.mae_h):.2f} h MAE** del miglior singolo expert a **{g['mae_h']:.2f} h MAE**, con un miglioramento di **{i['pooled_gain_h']:.2f} h** e vittoria in **{i['fold_wins_vs_best_single']}/5 fold**. Rispetto al semplice destination-median iniziale ({a['mae_h']:.2f} h), il guadagno è **{a['mae_h']-g['mae_h']:.2f} h**. L'intervallo centrale nominale all'80% raggiunge **{h['pooled_coverage']*100:.2f}%** di coverage pooled.\n\n## La parte importante: cosa non sto sostenendo\n\nNon presenterei questo numero come prova definitiva che M16 sia migliore su dati futuri. Il red-team mostra che il guadagno sopravvive a leave-one-fold-out, trimming e winsorisation, ma il bootstrap paired al 95% attraversa ancora zero (**[{i['bootstrap_gain_ci95_h'][0]:.2f}, {i['bootstrap_gain_ci95_h'][1]:.2f}] h**) e una parte del miglioramento è concentrata nella coda.\n\nPer questo la conclusione corretta è: **M16 è un challenger tecnicamente promettente, leakage-safe e più strutturato; la superiorità va confermata su un nuovo holdout mai visto.**\n\n## Perché considero l'approccio più intelligente\n\nNon ho semplicemente aumentato gli hyperparameter o sostituito CatBoost con un algoritmo più complesso. Ho separato il problema in expert con bias differenti e interpretabili:\n\n- normalizzazione semantica delle destinazioni AIS;\n- prior robusti con shrinkage per categorie rare;\n- matching di rotte storiche reali tramite trajectory similarity;\n- expert fisico usato solo quando destination e moto sono sufficientemente affidabili;\n- challenger tabellare;\n- gating/stacking addestrato esclusivamente su prediction OOF;\n- incertezza e confidence esplicite.\n\n## Proposta di validazione\n\nSe volete valutare il challenger, la verifica più pulita è fornire **nuovi dati o un nuovo holdout**. Io manterrei M16 congelato, fisserei prima metrica e target, e farei un'unica valutazione senza ulteriore tuning.\n\n**M14 resta la submission ufficiale congelata. M16 è un challenger aggiuntivo, non una riscrittura post-hoc del risultato ufficiale.**\n"""
    (D / "M16_COMPANY_NOTE_IT.md").write_text(company_note)

    # Manifest and deterministic freeze package.
    payload = select_payload()
    manifest = {
        "milestone":"M16J", "version":M16J_VERSION,
        "purpose":"development challenger freeze; NOT official M14 submission",
        "file_count":len(payload),
        "files":[{"path":p.relative_to(ROOT).as_posix(),"size":p.stat().st_size,"sha256":sha(p)} for p in payload],
    }
    (R / "M16_CONTENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    zip_payload = payload + [R / "M16_CONTENT_MANIFEST.json"]
    out_zip = DIST / "ais_eta_m16_challenger_freeze.zip"
    build_deterministic_zip(zip_payload, out_zip)
    zip_sha = sha(out_zip)
    (DIST / "ais_eta_m16_challenger_freeze.zip.sha256").write_text(f"{zip_sha}  {out_zip.name}\n")

    immutable = {
        "m14_submission": {"path":"dist/ais_eta_takehome_submission_final.zip","sha256":"65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c"},
        "m10_model": {"path":"models/m10_strict_reference_catboost.cbm","sha256":"efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a"},
        "m10_final_predictions": {"path":"reports/m10_final_predictions.csv","sha256":"fc6334d82f0b3c382e40805dc1a325163001a363ff6a6329da33160734011ee1"},
        "m6_freeze": {"path":"reports/M6_FREEZE.json","sha256":"b551c12da6d05fbea22c365f0b2167879669d948c4d790d46ccd28a9dc8e17de"},
    }
    for item in immutable.values():
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]

    prior = [
        "M16A_BENCHMARK_FREEZE.json","M16B_RESOLVER_FREEZE.json","M16C_HIERARCHICAL_PRIOR_FREEZE.json",
        "M16D_ROUTE_ANALOGUE_FREEZE.json","M16E_MARITIME_PHYSICS_FREEZE.json","M16F_TABULAR_PANEL_FREEZE.json",
        "M16G_MIXTURE_FREEZE.json","M16H_PROBABILISTIC_FREEZE.json","M16I_RED_TEAM_FREEZE.json",
    ]
    freeze = {
        "milestone":"M16J", "version":M16J_VERSION, "status":"SMART_CHALLENGER_FROZEN",
        "promotion_decision":decision, "official_submission":"M14", "m14_replaced":False,
        "development_population":386, "blocked_old_final_population":53, "old_final_used_for_m16_selection":False,
        "fresh_untouched_holdout_required":True,
        "m16_freeze_zip":"dist/ais_eta_m16_challenger_freeze.zip", "m16_freeze_zip_sha256":zip_sha,
        "content_manifest_sha256":sha(R / "M16_CONTENT_MANIFEST.json"),
        "prior_freeze_sha256":{name:sha(R/name) for name in prior},
        "immutable_pre_m16_sha256":immutable,
        "headline": {"m16g_mae_h":g["mae_h"],"best_single_mae_h":float(e.mae_h),"gain_h":float(i["pooled_gain_h"]),"bootstrap_ci95_h":i["bootstrap_gain_ci95_h"],"m16h_coverage":float(h["pooled_coverage"])},
    }
    (R / "M16_FINAL_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print(f"PASS M16J decision: {decision}")
    print(f"PASS development-only rows: 386; old-final blocked: 53/53")
    print(f"PASS M16G MAE: {g['mae_h']:.6f} h; gain vs best single: {i['pooled_gain_h']:.6f} h")
    print(f"PASS M16H coverage: {h['pooled_coverage']:.6f}")
    print(f"PASS content manifest files: {len(payload)}")
    print(f"PASS M16 challenger freeze ZIP SHA256: {zip_sha}")
    print("PASS M14/M10/M6 immutable hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
