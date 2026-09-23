#!/usr/bin/env python3
"""Build M16I red-team, ablation and stress-test artifacts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16a import extended_metrics
from ais_eta.m16d import RouteConfig, candidate_configs as route_configs, predict_route_analogue
from ais_eta.m16i import (
    M16I_BOOTSTRAP_REPS, M16I_BOOTSTRAP_SEED, M16I_VERSION,
    mae, paired_bootstrap_gain, red_team_gate, selected_path_counterfactual, support_bucket,
)
R = ROOT / "reports"


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def load_frame() -> pd.DataFrame:
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    h = pd.read_csv(R / "m16h_oof_intervals.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    cols_h = ["mmsi","reference_eta_status","canonical_destination","confidence_tier","physics_eligible","resolution_confidence","m16d_gate_used","p10_h","p50_h","p90_h","covered_80","interval_half_width_h","predicted_abs_error_h"]
    cols_c = ["mmsi","ship_type_cat","m16c_deepest_support","m16c_fallback_count"]
    cols_d = ["mmsi","destination_norm","m16d_neighbour_count","m16d_nearest_distance","m16d_median_neighbour_distance","pred_m16d_history_aggregate_catboost_h","pred_m16a_raw_destination_median_h","pred_m16a_catboost_current_snapshot_h"]
    cols_e = ["mmsi","pred_m16e_physics_h","physics_eligible","resolution_confidence"]
    x = g.merge(h[cols_h], on="mmsi", how="left", suffixes=("", "_h"))
    x = x.merge(c[cols_c], on="mmsi", how="left").merge(d[cols_d], on="mmsi", how="left").merge(e[cols_e], on="mmsi", how="left", suffixes=("", "_e"))
    if len(x) != 386 or x.mmsi.nunique() != 386: raise AssertionError("M16I development cardinality mismatch")
    return x.sort_values("mmsi", kind="mergesort").reset_index(drop=True)


def global_route_predictions(x: pd.DataFrame) -> np.ndarray:
    """Recompute route analogue with the frozen representation/k but no destination gate."""
    matrices = dict(np.load(R / "m16d_route_distance_matrices.npz"))
    selected = pd.read_csv(R / "m16d_outer_selected_configs.csv")
    cfg_map = {c.config_id: c for c in route_configs()}
    y = x.target_tte_h.to_numpy(float); folds = x.m16a_outer_fold.to_numpy(int)
    dest = np.asarray(["GLOBAL"] * len(x), dtype=object)
    out = np.full(len(x), np.nan, dtype=float)
    for outer in range(5):
        tr = np.flatnonzero(folds != outer); va = np.flatnonzero(folds == outer)
        cid = str(selected.loc[selected.outer_fold.eq(outer), "selected_config_id"].iloc[0])
        base = cfg_map[cid]
        cfg = RouteConfig(base.representation, "global", base.k)
        p = predict_route_analogue(train_indices=tr, valid_indices=va, distance_matrix=matrices[base.representation], targets=y, destinations=dest, config=cfg)
        out[va] = p.sort_values("row_index").prediction_h.to_numpy(float)
    if not np.isfinite(out).all(): raise AssertionError("global route ablation non-finite")
    return out


def structural_ablations(x: pd.DataFrame) -> pd.DataFrame:
    meta = x.m16g_selected_meta_id.astype(str).tolist(); sel = x.m16g_selected_expert.astype(str).tolist()
    prior = x.pred_m16c_prior_h.to_numpy(float); route = x.pred_m16d_route_h.to_numpy(float)
    physics_gate = x.pred_m16e_physics_gate_h.to_numpy(float); tab = x.pred_m16f_tabular_h.to_numpy(float)
    raw_phys = x.pred_m16e_physics_h.to_numpy(float); eligible = x.physics_eligible.astype(bool).to_numpy()
    raw_dest = x.pred_m16a_raw_destination_median_h.to_numpy(float); current = x.pred_m16a_catboost_current_snapshot_h.to_numpy(float)
    global_route = global_route_predictions(x)

    def rows(**arrs):
        return [{k: float(v[i]) for k,v in arrs.items()} for i in range(len(x))]

    no_phys = selected_path_counterfactual(meta, sel, rows(prior=prior, route=route, tabular=tab), ["prior","route","tabular"])
    phys_no_route = np.where(eligible & np.isfinite(raw_phys), raw_phys, prior)
    no_route = selected_path_counterfactual(meta, sel,
        rows(prior=prior, physics_gate=phys_no_route, tabular=tab), ["prior","physics_gate","tabular"])
    no_canonical = selected_path_counterfactual(meta, sel,
        rows(prior=raw_dest, route=global_route, physics_gate=global_route, tabular=current), ["prior","route","tabular"])
    no_history = selected_path_counterfactual(meta, sel,
        rows(prior=prior, route=current, physics_gate=current, tabular=current), ["prior","tabular"])

    preds = {
        "official_m16g": x.pred_m16g_selected_h.to_numpy(float),
        "best_single_m16e": x.pred_m16e_physics_gate_h.to_numpy(float),
        "no_physics_structural": no_phys,
        "no_route_structural": no_route,
        "no_canonical_destination_proxy": no_canonical,
        "no_longitudinal_history_proxy": no_history,
    }
    y = x.target_tte_h.to_numpy(float)
    rows_out=[]
    official_mae=mae(y,preds["official_m16g"])
    for name,p in preds.items():
        m=extended_metrics(y,p)
        rows_out.append({"scenario":name,"diagnostic_only":name not in {"official_m16g","best_single_m16e"},"mae_delta_vs_official_h":float(m["mae_h"]-official_mae),**m})
    return pd.DataFrame(rows_out), preds, global_route


def group_eval(x: pd.DataFrame, group: pd.Series, group_name: str) -> pd.DataFrame:
    y=x.target_tte_h.to_numpy(float); p=x.pred_m16g_selected_h.to_numpy(float); b=x.pred_m16e_physics_gate_h.to_numpy(float)
    lo=x.p10_h.to_numpy(float); hi=x.p90_h.to_numpy(float)
    out=[]
    for v in sorted(pd.Series(group).fillna("UNKNOWN").astype(str).unique()):
        m=pd.Series(group).fillna("UNKNOWN").astype(str).eq(v).to_numpy(); n=int(m.sum())
        if n==0: continue
        ae=np.abs(y[m]-p[m]); abe=np.abs(y[m]-b[m]); covered=((y[m]>=lo[m])&(y[m]<=hi[m]))
        out.append({"group":group_name,"value":v,"n":n,"m16g_mae_h":float(ae.mean()),"best_single_mae_h":float(abe.mean()),"gain_h":float((abe-ae).mean()),"m16g_medae_h":float(np.median(ae)),"m16g_p90_ae_h":float(np.quantile(ae,.9)),"coverage_80":float(covered.mean()),"mean_interval_width_h":float(np.mean(hi[m]-lo[m]))})
    return pd.DataFrame(out)


def main() -> int:
    x=load_frame()
    manifest=json.loads((R/"M16A_DEVELOPMENT_MANIFEST.json").read_text())
    blocked=set(int(z) for z in manifest["blocked_old_final_mmsi"])
    if set(x.mmsi.astype(int)) & blocked: raise AssertionError("old final leaked into M16I")
    y=x.target_tte_h.to_numpy(float); p=x.pred_m16g_selected_h.to_numpy(float); b=x.pred_m16e_physics_gate_h.to_numpy(float)
    ae=np.abs(y-p); aeb=np.abs(y-b); gain=aeb-ae; folds=x.m16a_outer_fold.to_numpy(int)

    # Outer-train destination support is computed without validation-row target use.
    supports=[]
    for i,row in x.iterrows():
        tr=x.loc[x.m16a_outer_fold.ne(int(row.m16a_outer_fold)),"canonical_destination"].astype(str)
        supports.append(int((tr==str(row.canonical_destination)).sum()))
    x["outer_train_destination_support"]=supports
    x["destination_support_bucket"]=[support_bucket(v) for v in supports]
    x["destination_semantic_bucket"]=np.where(x.canonical_destination.eq("FOR_ORDERS"),"FOR_ORDERS",
        np.where(x.canonical_destination.eq("UNKNOWN"),"UNKNOWN",
        np.where(x.canonical_destination.astype(str).str.startswith("AMBIGUOUS:"),"AMBIGUOUS","SPECIFIC_OR_OTHER")))
    x["route_support_bucket"]=np.where(x.m16d_neighbour_count<=5,"LOW_K_LE5", "HIGH_K_GT5")

    ablations,preds,global_route=structural_ablations(x)
    x["pred_m16i_global_route_no_destination_gate_h"]=global_route

    # Fold / leave-one-fold stability.
    fold_rows=[]
    for f in range(5):
        m=folds==f
        fold_rows.append({"outer_fold":f,"n":int(m.sum()),"m16g_mae_h":float(ae[m].mean()),"best_single_mae_h":float(aeb[m].mean()),"gain_h":float(gain[m].mean()),"m16g_wins":bool(gain[m].mean()>0)})
    fold_df=pd.DataFrame(fold_rows)
    loo=[]
    for f in range(5):
        m=folds!=f
        loo.append({"left_out_fold":f,"n":int(m.sum()),"gain_h":float(gain[m].mean()),"positive":bool(gain[m].mean()>0)})
    loo_df=pd.DataFrame(loo)

    # Robust trimming uses target magnitude, independent of model-difference direction.
    trim_rows=[]
    order=np.argsort(-np.abs(y))
    for frac in (0.0,.01,.025,.05,.10):
        k=int(np.floor(len(y)*frac)); mask=np.ones(len(y),bool)
        if k: mask[order[:k]]=False
        trim_rows.append({"trim_fraction_by_abs_target":frac,"removed_rows":k,"remaining_rows":int(mask.sum()),"m16g_mae_h":float(ae[mask].mean()),"best_single_mae_h":float(aeb[mask].mean()),"gain_h":float(gain[mask].mean())})
    trim_df=pd.DataFrame(trim_rows)
    wins_rows=[]
    both=np.concatenate([ae,aeb])
    for q in (.90,.95,.99):
        cap=float(np.quantile(both,q)); wa=np.minimum(ae,cap); wb=np.minimum(aeb,cap)
        wins_rows.append({"winsor_quantile":q,"cap_h":cap,"m16g_mae_h":float(wa.mean()),"best_single_mae_h":float(wb.mean()),"gain_h":float((wb-wa).mean())})
    wins_df=pd.DataFrame(wins_rows)

    # Paired stratified bootstrap on the frozen OOF per-row gain.
    boot=paired_bootstrap_gain(gain,folds,reps=M16I_BOOTSTRAP_REPS,seed=M16I_BOOTSTRAP_SEED)
    boot_df=pd.DataFrame({"replicate":np.arange(len(boot),dtype=int),"gain_h":boot})
    ci=np.quantile(boot,[.025,.05,.5,.95,.975])

    # Destination sensitivity: remove every destination with >=5 rows plus largest-net contributor.
    dest_net=pd.DataFrame({"dest":x.canonical_destination.astype(str),"gain":gain}).groupby("dest").agg(n=("gain","size"),net_gain_h=("gain","sum"),mean_gain_h=("gain","mean")).reset_index()
    top_net=str(dest_net.sort_values("net_gain_h",ascending=False,kind="mergesort").iloc[0].dest)
    dest_remove=[]
    candidates=sorted(set(dest_net.loc[dest_net.n>=5,"dest"].astype(str))|{top_net})
    for dest in candidates:
        m=x.canonical_destination.astype(str).ne(dest).to_numpy()
        dest_remove.append({"removed_destination":dest,"removed_n":int((~m).sum()),"remaining_gain_h":float(gain[m].mean()),"positive":bool(gain[m].mean()>0)})
    dest_remove_df=pd.DataFrame(dest_remove).sort_values(["remaining_gain_h","removed_destination"],kind="mergesort")

    # Error and gain concentration.
    concentration=[]
    n=len(x)
    for name,vals in (("m16g_abs_error",ae),("best_single_abs_error",aeb)):
        total=float(vals.sum())
        for k in (1,5,10,int(np.ceil(.05*n)),int(np.ceil(.10*n))):
            k=min(k,n); s=float(np.sort(vals)[::-1][:k].sum())
            concentration.append({"measure":name,"top_k":k,"share":s/total if total else 0.0})
    positive=np.sort(gain[gain>0])[::-1]; pos_total=float(positive.sum())
    for k in (1,3,5,10,20):
        k=min(k,len(positive)); concentration.append({"measure":"positive_gain","top_k":k,"share":float(positive[:k].sum()/pos_total) if pos_total else 0.0})
    concentration_df=pd.DataFrame(concentration)

    # Required subgroup stress diagnostics.
    vessel=x.ship_type_cat.fillna("UNKNOWN").astype(str)
    vc=vessel.value_counts(); vessel_group=vessel.where(vessel.map(vc)>=5,"OTHER_LT5")
    stress=pd.concat([
        group_eval(x,x.destination_support_bucket,"destination_support"),
        group_eval(x,x.destination_semantic_bucket,"destination_semantics"),
        group_eval(x,vessel_group,"vessel_category"),
        group_eval(x,x.reference_eta_status,"reference_eta_status_DIAGNOSTIC_ONLY"),
        group_eval(x,x.route_support_bucket,"route_support"),
        group_eval(x,x.m16d_gate_used,"route_gate"),
        group_eval(x,x.confidence_tier,"confidence_tier"),
    ],ignore_index=True)

    # Interval calibration curve by predicted-risk quintile and pathologies.
    risk_bin=pd.qcut(x.predicted_abs_error_h.rank(method="first"),5,labels=["Q1_LOW","Q2","Q3","Q4","Q5_HIGH"])
    calibration=group_eval(x,risk_bin.astype(str),"predicted_error_risk_quintile")
    calibration["coverage_gap_vs_nominal"] = calibration.coverage_80 - .80

    pathology_rows=[
        {"check":"nonfinite_point_predictions","count":int((~np.isfinite(p)).sum())},
        {"check":"nonfinite_interval_endpoints","count":int((~np.isfinite(x[["p10_h","p90_h"]].to_numpy(float))).sum())},
        {"check":"negative_point_prediction","count":int((p<0).sum())},
        {"check":"point_prediction_gt_30d","count":int((p>720).sum())},
        {"check":"point_prediction_lt_minus_7d","count":int((p<-168).sum())},
        {"check":"interval_width_gt_30d","count":int(((x.p90_h-x.p10_h)>720).sum())},
        {"check":"route_destination_fallback_global","count":int(x.m16d_gate_used.eq("destination_fallback_global").sum())},
        {"check":"route_global","count":int(x.m16d_gate_used.eq("global").sum())},
        {"check":"physics_not_eligible","count":int((~x.physics_eligible.astype(bool)).sum())},
    ]
    pathologies=pd.DataFrame(pathology_rows)

    pooled_gain=float(gain.mean()); fold_wins=int((fold_df.gain_h>0).sum()); loo_pos=int(loo_df.positive.sum())
    trim5=float(trim_df.loc[np.isclose(trim_df.trim_fraction_by_abs_target,.05),"gain_h"].iloc[0])
    win95=float(wins_df.loc[np.isclose(wins_df.winsor_quantile,.95),"gain_h"].iloc[0])
    leave_top=float(dest_remove_df.loc[dest_remove_df.removed_destination.eq(top_net),"remaining_gain_h"].iloc[0])
    pos_prob=float(np.mean(boot>0)); top5share=float(concentration_df.loc[(concentration_df.measure.eq("positive_gain"))&(concentration_df.top_k.eq(5)),"share"].iloc[0])
    conf=x.groupby("confidence_tier").apply(lambda g: float(np.median(np.abs(g.target_tte_h-g.pred_m16g_selected_h))), include_groups=False).to_dict()
    ordered=bool(conf["HIGH"]<=conf["MEDIUM"]<=conf["LOW"])
    coverage=float(np.mean((y>=x.p10_h)&(y<=x.p90_h)))
    gate=red_team_gate(pooled_gain_h=pooled_gain,fold_wins=fold_wins,leave_one_fold_positive=loo_pos,trimmed_5pct_gain_h=trim5,winsorized_95_gain_h=win95,leave_top_destination_gain_h=leave_top,bootstrap_positive_probability=pos_prob,bootstrap_ci95_low_h=float(ci[0]),top5_positive_gain_share=top5share,m16h_coverage=coverage,confidence_ordered=ordered)

    # Save row-level audit ledger.
    audit=x[["mmsi","m16a_outer_fold","target_tte_h","canonical_destination","destination_norm","ship_type_cat","reference_eta_status","confidence_tier","physics_eligible","m16d_gate_used","m16d_neighbour_count","outer_train_destination_support","destination_support_bucket","pred_m16g_selected_h","pred_m16e_physics_gate_h","p10_h","p50_h","p90_h","covered_80"]].copy()
    audit["ae_m16g_h"]=ae; audit["ae_best_single_h"]=aeb; audit["gain_h_vs_best_single"]=gain
    for name,pred in preds.items(): audit[f"pred_{name}_h"]=pred
    audit["pred_global_route_no_destination_gate_h"]=global_route

    # Deterministic visuals.
    fig,ax=plt.subplots(figsize=(9,5.5));
    plot=ablations.sort_values("mae_h",ascending=True); ax.barh(plot.scenario,plot.mae_h); ax.invert_yaxis(); ax.set_xlabel("OOF MAE (hours)"); ax.set_title("M16I structural ablation stress diagnostics")
    for i,v in enumerate(plot.mae_h): ax.text(float(v)+1,i,f"{v:.1f}",va="center")
    fig.tight_layout(); fig.savefig(R/"m16i_ablation_comparison.png",dpi=160,metadata={"Software":"AIS ETA M16I"}); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.2)); ax.bar(fold_df.outer_fold.astype(str),fold_df.gain_h); ax.axhline(0,linestyle="--",linewidth=1); ax.set_xlabel("Frozen outer fold"); ax.set_ylabel("MAE gain vs M16E (h)"); ax.set_title("M16I fold sensitivity of M16G gain"); fig.tight_layout(); fig.savefig(R/"m16i_fold_gain.png",dpi=160,metadata={"Software":"AIS ETA M16I"}); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.2)); ax.hist(boot,bins=40); ax.axvline(0,linestyle="--",linewidth=1); ax.set_xlabel("Paired bootstrap MAE gain (h)"); ax.set_ylabel("Replicates"); ax.set_title("M16I stratified paired bootstrap gain"); fig.tight_layout(); fig.savefig(R/"m16i_bootstrap_gain.png",dpi=160,metadata={"Software":"AIS ETA M16I"}); plt.close(fig)

    files={
      "m16i_ablation_metrics.csv":ablations,"m16i_fold_stability.csv":fold_df,"m16i_leave_one_fold_out.csv":loo_df,
      "m16i_trimmed_stress.csv":trim_df,"m16i_winsorized_stress.csv":wins_df,"m16i_bootstrap_gain.csv":boot_df,
      "m16i_destination_leave_one_out.csv":dest_remove_df,"m16i_error_concentration.csv":concentration_df,
      "m16i_subgroup_stress.csv":stress,"m16i_calibration_audit.csv":calibration,"m16i_prediction_pathologies.csv":pathologies,
      "m16i_row_audit.csv":audit,
    }
    for name,df in files.items(): df.to_csv(R/name,index=False,float_format="%.12g")

    summary={
      "milestone":"M16I","version":M16I_VERSION,"gate":gate,"development_rows":386,"blocked_old_final_rows":53,"final_test_used":False,
      "official_m16g_mae_h":float(ae.mean()),"best_single_m16e_mae_h":float(aeb.mean()),"pooled_gain_h":pooled_gain,"fold_wins_vs_best_single":fold_wins,
      "leave_one_fold_positive":loo_pos,"trimmed_5pct_abs_target_gain_h":trim5,"winsorized_95_gain_h":win95,
      "largest_net_gain_destination":top_net,"gain_after_removing_largest_net_destination_h":leave_top,
      "bootstrap_reps":M16I_BOOTSTRAP_REPS,"bootstrap_seed":M16I_BOOTSTRAP_SEED,"bootstrap_positive_probability":pos_prob,
      "bootstrap_gain_ci95_h":[float(ci[0]),float(ci[4])],"bootstrap_gain_ci90_h":[float(ci[1]),float(ci[3])],
      "top5_positive_gain_share":top5share,"m16h_coverage":coverage,"confidence_medae_h":conf,"confidence_ordered":ordered,
      "structural_ablation_note":"counterfactual path stresses preserve frozen M16G path where possible and use deterministic fallbacks; they are not retrained/reselected challenger models",
      "headline_caveat":"The pooled M16G gain survives fold omission, target-magnitude trimming, winsorisation and removal of the largest net-gain destination, but the 95% paired-bootstrap interval crosses zero and positive gain is concentrated in a few rows. Treat M16G as a promising development challenger, not proven replacement, until a new untouched holdout is scored.",
    }
    (R/"M16I_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

    report=f"""# M16I — Red-team, Ablation and Stress Testing\n\n## Decision\n\n**{gate}**\n\nM16I does not tune M16G/M16H. It audits the frozen 386-row development OOF ledger and keeps all 53 old M10 final MMSIs blocked.\n\n## Robustness of the M16G point gain\n\n- M16G MAE: **{ae.mean():.2f} h**\n- best single M16E MAE: **{aeb.mean():.2f} h**\n- pooled gain: **{pooled_gain:.2f} h**\n- fold wins: **{fold_wins}/5**; leave-one-fold-out gain remains positive in **{loo_pos}/5** cases\n- after removing the top 5% of rows by absolute target magnitude: **{trim5:.2f} h** gain\n- 95% winsorised paired gain: **{win95:.2f} h**\n- after removing largest net-gain destination `{top_net}`: **{leave_top:.2f} h** gain\n- stratified paired bootstrap P(gain>0): **{pos_prob:.1%}**; 95% interval **[{ci[0]:.2f}, {ci[4]:.2f}] h**\n\nThe red-team therefore finds a real central robustness signal, but not a clean 95% statistical separation. Positive improvement is also concentrated: the five largest positive-gain rows account for **{top5share:.1%}** of total positive gain. This is preserved as a limitation, not hidden.\n\n## Structural ablations\n\nThe required `no physics`, `no route`, `no canonical destination`, and `no longitudinal history` tests are deterministic counterfactual path stresses. They do **not** retrain or reselect a challenger after observing red-team results. Therefore they answer dependency/sensitivity questions, not 'what is the best model without component X?'. See `m16i_ablation_metrics.csv`.\n\n## Subgroup and uncertainty stress\n\nAudits cover cold/rare canonical destination support computed from each row's outer-train, `FOR_ORDERS`/unknown/ambiguous destination classes, vessel-category groups, target-derived stale/future status for diagnostics only, low route support, route fallback mode and M16H confidence tiers. Calibration is also checked across predicted-error-risk quintiles.\n\n## Interpretation\n\nM16G survives the fair robustness tests based on folds, target-magnitude trimming/winsorisation and destination removal, and M16H retains nominal pooled coverage. However, heavy tails remain dominant and the paired-bootstrap 95% interval crosses zero. The appropriate claim is therefore **development-only promising challenger with a heavy-tail caveat**, not proven superiority. A new untouched company holdout remains required for a fresh generalisation claim.\n"""
    (R/"M16I_REPORT.md").write_text(report)

    artifacts=["M16I_REPORT.md","M16I_SUMMARY.json",*files.keys(),"m16i_ablation_comparison.png","m16i_fold_gain.png","m16i_bootstrap_gain.png"]
    prior=["M16A_BENCHMARK_FREEZE.json","M16B_RESOLVER_FREEZE.json","M16C_HIERARCHICAL_PRIOR_FREEZE.json","M16D_ROUTE_ANALOGUE_FREEZE.json","M16E_MARITIME_PHYSICS_FREEZE.json","M16F_TABULAR_PANEL_FREEZE.json","M16G_MIXTURE_FREEZE.json","M16H_PROBABILISTIC_FREEZE.json"]
    freeze={"milestone":"M16I","version":M16I_VERSION,"gate":gate,"development_population":386,"blocked_old_final_population":53,"final_test_used":False,"red_team_only_no_tuning":True,"artifact_sha256":{n:sha(R/n) for n in artifacts},"prior_freeze_sha256":{n:sha(R/n) for n in prior}}
    (R/"M16I_RED_TEAM_FREEZE.json").write_text(json.dumps(freeze,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
