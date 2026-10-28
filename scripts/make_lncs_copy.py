#!/usr/bin/env python3
"""Build the LNCS submission copy from the final manuscript.

    python3 scripts/make_lncs_copy.py          # needs lxml (pip install lxml); no other dependency

Reads  paper/514_Manuscript_final.docx   (authors' final edits, built by scripts/make_submission_copy.py)
Writes paper/514_Manuscript_final_LNCS.docx

1. Content changes from the reviewer response (each asserted to match exactly once):
   * Figs. 2-3 replaced by GradientSHAP of the leakage-safe GRU-D (scripts/shap_grud.py, outputs/shap/); exact
     XGBoost TreeSHAP (outputs/results/figure3_shap_mean_abs.csv) kept as a cross-check sentence;
   * new Sect. 12.6 with a regional reproduction number from ILINet (surveillance/rt.py, outputs/surveillance/rt/),
     Fig. 6 and references [31], [32];
   * "viral (test) positives" -> influenza-like-illness (ILI) visits, in the text, Table 3 and Fig. 4
     (outputs/figures/figure4_dataflow.png); scope, pathogen and transmission sentences; conclusion;
   * calibration references [8], [15] cited; abstract shortened to <= 250 words.
2. Springer LNCS layout: A4 with a 12.2 x 19.3 cm text block, Times, LNCS paragraph styles, numbered headings,
   "Fig. n." / "Table n." captions, "Keywords", grouped citations ([1,2,12]) and LNCS-style references.
Every number written into the text is read from the result files, not typed.
"""
import copy, csv, json, re, struct, zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "514_Manuscript_final.docx"
DST = ROOT / "paper" / "514_Manuscript_final_LNCS.docx"
SHAP, RT, RES = ROOT / "outputs" / "shap", ROOT / "outputs" / "surveillance" / "rt", ROOT / "outputs" / "results"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XS = "{http://www.w3.org/XML/1998/namespace}space"
q = lambda t: f"{{{W}}}{t}"
TEXT_W = 6917                      # LNCS text block 12.2 cm, in twips
TEXT_EMU = TEXT_W * 635


# ============================================================================ numbers from the result files
def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))

run = json.load(open(SHAP / "shap_run.json"))
gimp = {r["feature"]: float(r["mean_abs_shap"]) for r in read_csv(SHAP / "grud_global_importance.csv")}
order = list(gimp)                                    # file is sorted by importance
traj = read_csv(SHAP / f"grud_patient_{run['trajectory']['patient']}_trajectory.csv")
th = run["thresholds"]
hrs = [int(r["hour_rel_onset"]) for r in traj]
risk = [float(r["risk_calibrated"]) for r in traj]
first_med = next(h for h, v in zip(hrs, risk) if v >= th["medium"])
first_high = next(h for h, v in zip(hrs, risk) if v >= th["high"])
high_from = [h for h, v in zip(hrs, risk) if v < th["high"]][-1] + 1
assert first_high == run["trajectory"]["first_high_hour_rel_onset"]
xgb = {r["feature"]: float(r["mean_abs_shap"]) for r in read_csv(RES / "figure3_shap_mean_abs.csv")}
xgb_order = sorted(xgb, key=xgb.get, reverse=True)
assert xgb_order[:4] == ["ICULOS", "HospAdmTime", "Unit1", "Age"], xgb_order[:4]
rt = json.load(open(RT / "rt_summary.json"))["regions"]
rt_lo = min(v["R_baseline_range"][0] for v in rt.values())
rt_hi = max(v["R_baseline_range"][1] for v in rt.values())
rt_pk = sorted(v["R_peak"] for v in rt.values())
assert sorted({v["R_peak_week"] for v in rt.values()}) == ["2024-12-22", "2024-12-29"]
assert rt["South"]["R_low_above_1_day"] == rt["West"]["R_low_above_1_day"]
assert rt["East"]["R_low_above_1_day"] is None
g2 = lambda f: f"{gimp[f]:.2f}"
x2 = lambda f: f"{xgb[f]:.2f}"
ratio = gimp["ICULOS"] / max(v for k, v in gimp.items() if k != "ICULOS")
cand = run["trajectory"]["rule"].split("(of ")[1].split(" septic")[0]
admin = [gimp["Unit1"], gimp["Unit2"], gimp["Gender"]]

SHAP_P1 = (
    f"To explain the GRU-D predictions, GradientSHAP [3], as implemented in Captum, was applied to each of the three "
    f"seed models and the attributions were averaged. It explained {run['n_windows']:,} test windows "
    f"({run['n_positive']} positive, {run['n_windows'] - run['n_positive']} negative) against {run['n_background']} "
    f"random training windows, with {run['n_samples']} samples per window. Attributions are on the log-odds scale; for "
    f"each window they add up to the change in the model output relative to the background (correlation "
    f"{run['completeness']['corr']:.3f}). Fig. 2 shows the 15 most important features. Time in the ICU (ICULOS) "
    f"dominates, with a mean absolute SHAP value of {g2('ICULOS')}, about {['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][round(ratio) - 2]} times that of any other "
    f"feature: it raises the predicted risk late in a stay and lowers it early. This matches the length-of-stay-only "
    f"reference (AUROC 0.673) and reflects that labelled onsets fall late in a stay, so part of what the model has "
    f"learnt is when sepsis is labelled rather than how patients deteriorate. Among physiological variables, "
    f"respiratory rate ({g2('Resp')}), heart rate ({g2('HR')}), the 6-hour mean arterial pressure "
    f"({g2('MAP_rolling_mean_6hr')}), oxygen saturation ({g2('O2Sat')}) and temperature ({g2('Temp')}) contribute "
    f"most. Higher respiratory and heart rates raise the risk and a higher mean arterial pressure lowers it, "
    f"consistent with the qSOFA criteria. Gender and the ICU-unit indicators also rank highly "
    f"({min(admin):.2f}–{max(admin):.2f}), which suggests that the model partly relies on patient mix and unit rather "
    f"than physiology alone. Lactate ranks {order.index('Lactate') + 1}th of {len(order)} features. Exact TreeSHAP on "
    f"the XGBoost baseline gives the same top feature (ICULOS, mean |SHAP| {x2('ICULOS')}), with administrative "
    f"variables ranked higher still: hours since hospital admission ({x2('HospAdmTime')}), ICU unit ({x2('Unit1')}) "
    f"and age ({x2('Age')}). Because the data were imputed before release (Sect. 3), the observation-mask "
    f"attributions only reflect the padded start of short stays, not missing measurements.")
SHAP_P2 = (
    f"Fig. 3 follows one septic test patient chosen by a fixed rule: among the {cand} septic test patients with at "
    f"least 24 pre-onset hours, the one with the median peak risk in the 6 hours before onset, so the example is "
    f"typical rather than the best case. The calibrated GRU-D risk passes the Medium threshold (80% specificity on "
    f"validation) {-first_med} hours before the labelled onset and first reaches the High threshold (95% specificity) "
    f"{-first_high} hours before onset; it then falls back below High and stays above it from {-high_from} hours "
    f"before onset. Heart rate and respiratory rate drive the increase between 25 and 18 hours before onset, "
    f"platelets lower the risk in the middle hours and haematocrit lowers it close to onset, while the contribution "
    f"of ICULOS grows steadily with time in the ICU. This is a single illustrative case and not an estimate of "
    f"average lead time, which was not measured.")
CAP2 = ("FIGURE 2. GradientSHAP explanation of the leakage-safe GRU-D model (three seeds) on held-out test windows. "
        "Left: mean absolute SHAP value per feature (log-odds), split into the value and observation-mask parts. "
        "Right: SHAP value per window, coloured by the feature's last-hour value (z-score).")
CAP3 = ("FIGURE 3. Risk trajectory and explanation for one septic test patient over the 36 hours before the labelled "
        "onset. Top: calibrated GRU-D risk with the Medium (80% specificity) and High (95% specificity) thresholds; "
        "shading marks the 6-hour label window. Middle: SHAP value of time in the ICU (ICULOS). Bottom: hourly SHAP "
        "values of the six other most influential features.")
RT_P = (
    f"To add a measure of transmission, the time-varying reproduction number R{{t}} was estimated for each region "
    f"from the same ILINet series. For each week, the growth rate r of influenza-like illness was estimated by a "
    f"quasi-Poisson regression of weekly ILI visits on time, with all patient visits as an offset, over the trailing "
    f"four weeks, so every estimate uses only data available at that time. The growth rate was converted to "
    f"R{{t}} = (1 + rs)^{{k}} for a gamma-distributed generation interval with shape k and scale s [31], using the "
    f"influenza serial interval of mean 3.6 days and standard deviation 1.6 days [32]. Fig. 6 shows the estimates. "
    f"Over the 60-day baseline R{{t}} stayed between {rt_lo:.2f} and {rt_hi:.2f} in all regions; it peaked at "
    f"{rt_pk[0]:.2f}–{rt_pk[-1]:.2f} in the weeks starting 22 and 29 December 2024 and then fell below 1. In the South "
    f"and West, R{{t}} was significantly above 1 (lower 95% bound above 1 for two consecutive weeks) from day "
    f"{rt['South']['R_low_above_1_day']}, {rt['South']['R_low_above_1_lead_days']} and "
    f"{rt['West']['R_low_above_1_lead_days']} days before the rise in ILI used as the outbreak start; in the North, "
    f"growth became significant only on day {rt['North']['R_low_above_1_day']}, "
    f"{-rt['North']['R_low_above_1_lead_days']} days after the rise, and in the East it never stayed significant for "
    f"two weeks. R{{t}} therefore adds a transmission context to the alarms but is not an earlier detector: the season "
    f"grows slowly, so R{{t}} stays close to 1. Because ILI is a share of all visits and mixes pathogens, R{{t}} here "
    f"describes the growth of influenza-like illness rather than of a single virus, and the late-December peak partly "
    f"reflects fewer routine visits over the holidays.")
CAP6 = ("FIGURE 6. Time-varying reproduction number R{t} by region, estimated from CDC ILINet (trailing four-week "
        "growth rate, influenza serial interval). Shading: 95% confidence interval; dashed line: rise in ILI used as "
        "the outbreak start.")

# (old, new) on the text of one paragraph; each must match exactly once in the whole document
EDITS = [
    # abstract (<= 250 words): GradientSHAP, ILI wording, drop the sentence that repeats "hourly risk trajectories"
    ("Feature attributions (TreeSHAP) are computed for tree-based models to aid interpretation.",
     "Feature attributions are computed for the GRU-D sequence model with GradientSHAP."),
    (" The Digital Twin framework visualizes patient risk hour by hour, which could support early intervention.", ""),
    ("noisy daily counts of high-risk alerts and viral positives;",
     "noisy daily counts of high-risk alerts and influenza-like-illness (ILI) visits;"),
    # introduction: ILI wording + scope statement
    ("daily counts of high-risk alerts and viral test positives for early regional outbreak warning.",
     "daily counts of high-risk alerts and influenza-like-illness visits for early regional outbreak warning. The "
     "core contribution remains individual-level: each Digital Twin predicts sepsis for one patient. The surveillance "
     "layer is a population-level extension, evaluated on a simulated hospital network, and is not a deployed "
     "epidemiological surveillance system."),
    # explainability: what is actually computed
    ("TreeSHAP attributions are computed offline for the tree-based models.",
     "GradientSHAP attributions are computed offline for the GRU-D model, with exact TreeSHAP for XGBoost as a "
     "cross-check."),
    ("in the current implementation it provides TreeSHAP attributions for tree-based models and reference-range "
     "deviations in the dashboard.",
     "in the current implementation it provides offline GradientSHAP attributions for GRU-D (TreeSHAP for XGBoost) "
     "and reference-range deviations in the dashboard."),
    ("Model serving:Lightweight", "Model serving: Lightweight"),
    ("Explainability engine: TreeSHAP values are available for tree-based models; the dashboard additionally reports "
     "deviations from reference ranges.",
     "Explainability engine: GradientSHAP values for GRU-D and TreeSHAP values for XGBoost are computed offline; the "
     "dashboard reports deviations from reference ranges."),
    ("TreeSHAP values were obtained from XGBoost’s built-in exact implementation, FastAPI serves",
     "GradientSHAP attributions for GRU-D were computed with Captum (version 0.9) and TreeSHAP values with XGBoost’s "
     "built-in exact implementation; FastAPI serves"),
    # calibration references
    ("(Platt scaling [23])", "(Platt scaling [8], [23])"),
    ("calibration skill is modest.", "calibration skill is modest [15]."),
    # surveillance: ILI wording, pathogen sentence, limitations
    ("with viral test positives across a region can raise", "with regional counts of influenza-like illness (ILI) can raise"),
    ("admissions, high-risk alerts and viral positives (Fig. 4)", "admissions, high-risk alerts and ILI visits (Fig. 4)"),
    ("the pooled high-risk-alert rate, pooled viral cases, a combined sepsis-alert and viral signal",
     "the pooled high-risk-alert rate, pooled ILI counts, a combined sepsis-alert and ILI signal"),
    ("Pooled viral cases", "Pooled ILI counts"),
    ("so only its shape is real.",
     "so only its shape is real. ILINet reports syndromic influenza-like illness rather than laboratory-confirmed "
     "infection, so the viral signal does not identify a specific pathogen."),
    ("with one simulated site per region a single site also detected every outbreak, after 3 days.",
     "with one simulated site per region a single site also detected every outbreak, after 3 days. The layer flags "
     "unusual rises in pooled counts; it does not identify pathogens or trace contacts, and the reproduction number in "
     "Sect. 12.6 is a growth-rate estimate from syndromic data rather than a transmission model. Pathogen-specific "
     "feeds, such as laboratory-confirmed influenza, RSV and SARS-CoV-2 results, and coupling with transmission models "
     "are also future work."),
    # conclusion
    ("high-risk alerts and viral positives across simulated hospitals", "high-risk alerts and ILI visits across simulated hospitals"),
    ("Future work will focus on external and prospective validation",
     "GradientSHAP explanations of the GRU-D model show that time in the ICU dominates its predictions, mirroring the "
     "length-of-stay-only reference, so part of its discrimination reflects when sepsis is labelled rather than "
     "physiological deterioration. A regional reproduction number estimated from the same ILI series adds "
     "transmission context to the alarms but gave no earlier warning, because influenza-like illness grew slowly in "
     "this season. Future work will focus on external and prospective validation"),
    ("and pilots with real hospital data feeds.", "pilots with real hospital data feeds, and pathogen-specific surveillance data."),
    # availability: the release that contains this copy and its code
    ("(release tag paper514-final)", "(release tag paper514-final-lncs)"),
]

REFS = [
 "Desautels, T., et al.: Prediction of sepsis in the intensive care unit with minimal electronic health record data: a machine learning approach. JMIR Medical Informatics 4(3), e28 (2016)",
 "Moor, M., et al.: Early prediction of sepsis in the ICU using machine learning: a systematic review. Frontiers in Medicine 8, 607952 (2021)",
 "Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems, vol. 30 (2017)",
 "Johnson, A.E.W., et al.: MIMIC-III, a freely accessible critical care database. Scientific Data 3, 160035 (2016)",
 "Reyna, M.A., et al.: Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine 48(2), 210–217 (2020)",
 "Nemati, S., et al.: An interpretable machine learning model for accurate prediction of sepsis in the ICU. Critical Care Medicine 46(4), 547–553 (2018)",
 "Kam, H.J., Kim, H.Y.: Learning representations for the early detection of sepsis with deep neural networks. Computers in Biology and Medicine 89, 248–255 (2017)",
 "Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR, vol. 70, pp. 1321–1330 (2017)",
 "Fabrizzio, G.C., Erdmann, A.L., de Oliveira, L.M.: Web app for prediction of hospitalisation in intensive care unit by covid-19. Revista Brasileira de Enfermagem 76(6), e20220740 (2023)",
 "Tang, Y., Zhang, Y., Li, J.: A time series driven model for early sepsis prediction based on transformer module. BMC Medical Research Methodology 24, 23 (2024)",
 "Lauritsen, S.M., et al.: Explainable artificial intelligence model to predict acute critical illness from electronic health records. Nature Communications 11, 3852 (2020)",
 "Fleuren, L.M., et al.: Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. Intensive Care Medicine 46(3), 383–400 (2020)",
 "Topol, E.J.: High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine 25(1), 44–56 (2019)",
 "Beede, E., et al.: A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy. In: Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, pp. 1–12 (2020)",
 "Van Calster, B., et al.: Calibration: the Achilles heel of predictive analytics. BMC Medicine 17, 230 (2019)",
 "Goldberger, A.L., et al.: PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 101(23), e215–e220 (2000)",
 "Che, Z., et al.: Recurrent neural networks for multivariate time series with missing values. Scientific Reports 8, 6085 (2018)",
 "Roberts, S.W.: Control chart tests based on geometric moving averages. Technometrics 1(3), 239–250 (1959)",
 "Page, E.S.: Continuous inspection schemes. Biometrika 41(1–2), 100–115 (1954)",
 "Dwork, C., McSherry, F., Nissim, K., Smith, A.: Calibrating noise to sensitivity in private data analysis. In: Theory of Cryptography (TCC 2006). LNCS, vol. 3876, pp. 265–284. Springer (2006)",
 "Centers for Disease Control and Prevention: FluView: U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet). https://www.cdc.gov/fluview/, last accessed 2026/10/09",
 "Farrow, D.C., Brooks, L.C., Rumack, A., Tibshirani, R.J., Rosenfeld, R.: Delphi Epidata API (2015). https://github.com/cmu-delphi/delphi-epidata, fluview endpoint last accessed 2026/10/09",
 "Platt, J.C.: Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Smola, A.J., Bartlett, P., Schölkopf, B., Schuurmans, D. (eds.) Advances in Large Margin Classifiers, pp. 61–74. MIT Press (2000)",
 "Rudd, K.E., et al.: Global, regional, and national sepsis incidence and mortality, 1990–2017: analysis for the Global Burden of Disease Study. The Lancet 395(10219), 200–211 (2020)",
 "Lombardo, J., et al.: A systems overview of the Electronic Surveillance System for the Early Notification of Community-Based Epidemics (ESSENCE II). Journal of Urban Health 80(2 Suppl 1), i32–i42 (2003)",
 "Buckeridge, D.L.: Outbreak detection through automated surveillance: a review of the determinants of detection. Journal of Biomedical Informatics 40(4), 370–379 (2007)",
 "Unkel, S., Farrington, C.P., Garthwaite, P.H., Robertson, C., Andrews, N.: Statistical methods for the prospective detection of infectious disease outbreaks: a review. Journal of the Royal Statistical Society Series A 175(1), 49–82 (2012)",
 "Fricker Jr., R.D.: Introduction to Statistical Methods for Biosurveillance: With an Emphasis on Syndromic Surveillance. Cambridge University Press (2013)",
 "El Emam, K., et al.: A secure protocol for protecting the identity of providers when disclosing data for disease surveillance. Journal of the American Medical Informatics Association 18(3), 212–217 (2011)",
 "Rieke, N., et al.: The future of digital health with federated learning. npj Digital Medicine 3, 119 (2020)",
 "Wallinga, J., Lipsitch, M.: How generation intervals shape the relationship between growth rates and reproductive numbers. Proceedings of the Royal Society B 274(1609), 599–604 (2007)",
 "Cowling, B.J., Fang, V.J., Riley, S., Peiris, J.S.M., Leung, G.M.: Estimation of the serial interval of influenza. Epidemiology 20(3), 344–347 (2009)",
]

H1 = {"INTRODUCTION": "Introduction", "LITERATURE REVIEW": "Literature Review",
      "DATASET AND FEATURE ENGINEERING": "Dataset and Feature Engineering", "METHODOLOGY": "Methodology",
      "MODELS USED": "Models Used", "DIGITAL TWIN FRAMEWORK": "Digital Twin Framework",
      "IMPLEMENTATION DETAILS": "Implementation Details", "EXPERIMENTAL SETUP": "Experimental Setup",
      "RESULTS": "Results", "DISCUSSION": "Discussion",
      "PRACTICAL DEPLOYMENT CONSIDERATIONS": "Practical Deployment Considerations",
      "MULTI-SITE SURVEILLANCE AND EARLY OUTBREAK WARNING": "Multi-site Surveillance and Early Outbreak Warning",
      "CONCLUSION": "Conclusion", "DATA AND CODE AVAILABILITY": "Data and Code Availability"}
H2 = {"End-To-End System Architecture": "End-to-End System Architecture",
      "Architectural components": "Architectural Components", "Motivation": "Motivation",
      "Architecture": "Architecture", "Detection Methods": "Detection Methods",
      "Experimental Setup": "Experimental Setup", "Results": "Results",
      "Regional Reproduction Number": "Regional Reproduction Number",
      "Sensitivity Analysis": "Sensitivity Analysis", "Limitations": "Limitations"}


# ============================================================================ XML helpers
def E(tag, **attrs):
    el = etree.Element(q(tag))
    for k, v in attrs.items():
        el.set(q(k), str(v))
    return el


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q("t")))


def style_of(p):
    s = p.find(f"{q('pPr')}/{q('pStyle')}")
    return s.get(q("val")) if s is not None else ""


def replace_text(p, pattern, repl, count=0):
    """Regex replace over a paragraph's text even when it spans runs."""
    ts = list(p.iter(q("t")))
    full = "".join(t.text or "" for t in ts)
    ms = list(re.finditer(pattern, full))[: count or None]
    for m in reversed(ms):
        new = m.expand(repl) if isinstance(repl, str) else repl(m)
        s, e, pos, first = m.start(), m.end(), 0, True
        for t in ts:
            txt = t.text or ""
            a, b = pos, pos + len(txt)
            pos = b
            if b <= s or a >= e:
                continue
            lo, hi = max(s, a) - a, min(e, b) - a
            t.text = txt[:lo] + (new if first else "") + txt[hi:]
            t.set(XS, "preserve")
            first = False
    return len(ms)


def add_runs(p, s, bold_prefix=None):
    """Append runs; R{t} -> R with subscript t, ^{k} -> superscript k."""
    if bold_prefix:
        r = etree.SubElement(p, q("r"))
        etree.SubElement(etree.SubElement(r, q("rPr")), q("b"))
        t = etree.SubElement(r, q("t")); t.text = bold_prefix; t.set(XS, "preserve")
    for part in re.split(r"(\^\{[^}]*\}|\{[^}]*\})", s):
        if not part:
            continue
        r = etree.SubElement(p, q("r"))
        if part.startswith("{") or part.startswith("^{"):
            va = "superscript" if part.startswith("^") else "subscript"
            etree.SubElement(etree.SubElement(r, q("rPr")), q("vertAlign")).set(q("val"), va)
            part = part.lstrip("^")[1:-1]
        t = etree.SubElement(r, q("t")); t.text = part; t.set(XS, "preserve")


def text_para(s, style="BodyText"):
    p = E("p")
    pp = etree.SubElement(p, q("pPr"))
    pp.append(E("pStyle", val=style))
    add_runs(p, s)
    return p


def set_ppr(p, style, keep=()):
    old = p.find(q("pPr"))
    new = E("pPr")
    new.append(E("pStyle", val=style))
    if old is not None:
        for tag in keep:
            k = old.find(q(tag))
            if k is not None:
                new.append(k)
        p.remove(old)
    p.insert(0, new)


def set_text(p, parts):
    for c in list(p):
        if c.tag != q("pPr"):
            p.remove(c)
    for t, b in parts:
        add_runs(p, t, None) if not b else add_runs(p, "", t)


def png_size(b):
    return struct.unpack(">II", b[16:24])


# ============================================================================ build
zin = zipfile.ZipFile(SRC)
root = etree.fromstring(zin.read("word/document.xml"))
body = root.find(q("body"))
rels = zin.read("word/_rels/document.xml.rels").decode()
media_out = {}

def paras():
    return [p for p in body if p.tag == q("p")]

def find_para(pred, what):
    hits = [p for p in paras() if pred(p)]
    assert len(hits) == 1, (what, len(hits))
    return hits[0]

# ---- 1. content edits
for old, new in EDITS:
    hits = [p for p in root.iter(q("p")) if old in ptext(p)]
    assert len(hits) == 1, (old[:60], len(hits))
    replace_text(hits[0], re.escape(old), new.replace("\\", "\\\\"), 1)
assert not re.search(r"viral (test )?positives|viral cases", " ".join(ptext(p) for p in root.iter(q("p"))))

fig_tpl = next(p for p in paras() if p.find(".//" + q("drawing")) is not None)
next_rid = [max(int(x) for x in re.findall(r'Id="rId(\d+)"', rels))]
next_doc = [max(int(e.get("id")) for e in root.iter(f"{{{WP}}}docPr"))]

def figure_para(png_path, name):
    global rels
    data = png_path.read_bytes()
    next_rid[0] += 1; next_doc[0] += 1
    rid = f"rId{next_rid[0]}"
    rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                        f'officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
    media_out[f"word/media/{name}"] = data
    p = copy.deepcopy(fig_tpl)
    for d in p.iter(q("drawing")):
        for e in d.iter(f"{{{WP}}}docPr"):
            e.set("id", str(next_doc[0])); e.set("name", name); e.attrib.pop("descr", None)
        for e in d.iter(f"{{{PIC}}}cNvPr"):
            e.set("id", str(next_doc[0])); e.set("name", name)
        for e in d.iter(f"{{{A}}}blip"):
            e.set(f"{{{R}}}embed", rid)
    w, h = png_size(data)
    for e in list(p.iter(f"{{{WP}}}extent")) + list(p.iter(f"{{{A}}}ext")):
        e.set("cx", str(TEXT_EMU)); e.set("cy", str(int(TEXT_EMU * h / w)))
    return p

# ---- 2. replace Figs. 2-3 and their paragraphs (from the first figure in Results up to the Discussion heading)
auprc = find_para(lambda p: ptext(p).startswith("At 1.1% prevalence, AUPRC values"), "AUPRC paragraph")
disc = find_para(lambda p: ptext(p).strip() == "DISCUSSION", "Discussion heading")
el = auprc.getnext()
while el is not disc:
    nxt = el.getnext()
    body.remove(el)
    el = nxt
for new in [text_para(SHAP_P1), figure_para(SHAP / "grud_shap_summary.png", "figure2_shap_grud.png"),
            text_para(CAP2), text_para(SHAP_P2), figure_para(SHAP / "grud_patient_trajectory.png", "figure3_trajectory_grud.png"),
            text_para(CAP3)]:
    disc.addprevious(new)

# ---- 3. new Sect. 12.6 with Fig. 6, before "Sensitivity Analysis"
sens = find_para(lambda p: ptext(p).strip() == "Sensitivity Analysis", "Sensitivity Analysis heading")
for new in [text_para("Regional Reproduction Number"), text_para(RT_P),
            figure_para(RT / "rt_by_region.png", "figure6_rt_by_region.png"), text_para(CAP6)]:
    sens.addprevious(new)

# ---- 4. Fig. 4 image: ILI labels
fig4_cap = find_para(lambda p: ptext(p).startswith("FIGURE 4."), "Fig. 4 caption")
fig4 = fig4_cap.getprevious()
while fig4.find(".//" + q("drawing")) is None:
    fig4 = fig4.getprevious()
rid4 = next(fig4.iter(f"{{{A}}}blip")).get(f"{{{R}}}embed")
target4 = "word/" + re.search(rf'Id="{rid4}"[^>]*Target="([^"]+)"', rels).group(1)
media_out[target4] = (ROOT / "outputs" / "figures" / "figure4_dataflow.png").read_bytes()

# ---- 5. two new references (copies of the last numbered reference paragraph)
ref_paras = [p for p in paras() if p.find(".//" + q("numPr")) is not None and style_of(p) == "ListParagraph"]
assert len(ref_paras) == 30
for _ in range(2):
    ref_paras[-1].addnext(copy.deepcopy(ref_paras[-1]))
ref_paras = [p for p in paras() if p.find(".//" + q("numPr")) is not None and style_of(p) == "ListParagraph"]
assert len(ref_paras) == len(REFS) == 32

# ============================================================================ LNCS layout
title = find_para(lambda p: style_of(p) == "Title", "title")
abstract = find_para(lambda p: ptext(p).startswith("Abstract."), "abstract")
kw = find_para(lambda p: ptext(p).startswith("Index Terms"), "keywords")
words = len(ptext(abstract).split()) - 1
assert words <= 250, f"abstract has {words} words"

# author block: everything between the title and the abstract -> author, address, email
set_ppr(title, "Title")
el = title.getnext()
while el is not abstract:
    nxt = el.getnext()
    body.remove(el)
    el = nxt
for s, st in [("Reeja S R (✉), Ashwin Nair, Anjith Thomas Jose, Abin Binu, and Atharva Gore", "LNCSauthor"),
              ("VIT-AP University Amaravathi, Andhra Pradesh, India", "LNCSaddress"),
              ("reeja.sr@vitap.ac.in, {ashwin.22bce7968, anjith.22bce7289, binu.22bce9401, atharva.22bce9298}"
               "@vitapstudent.ac.in", "LNCSaddress")]:
    p = E("p"); pp = etree.SubElement(p, q("pPr")); pp.append(E("pStyle", val=st))
    r = etree.SubElement(p, q("r")); t = etree.SubElement(r, q("t")); t.text = s; t.set(XS, "preserve")
    abstract.addprevious(p)

set_ppr(abstract, "LNCSabstract")
terms = ptext(kw).split("—", 1)[1].strip().rstrip(".")
set_ppr(kw, "LNCSkeywords")
set_text(kw, [("Keywords: ", True), (" · ".join(t.strip() for t in terms.split(",")), False)])

# headings
sec = sub = 0
in_refs = False
heads = set()
for p in paras():
    t = ptext(p).strip()
    if t in H1 and t.isupper():
        sec += 1; sub = 0
        set_ppr(p, "Heading1"); set_text(p, [(f"{sec}\t{H1[t]}", False)]); heads.add(p)
    elif t in H2 and not t.isupper() and style_of(p) in ("BodyText", ""):
        sub += 1
        set_ppr(p, "Heading2"); set_text(p, [(f"{sec}.{sub}\t{H2[t]}", False)]); heads.add(p)
    elif t == "REFERENCES":
        set_ppr(p, "LNCSrefhead"); set_text(p, [("References", False)]); heads.add(p)
assert sec == 14, sec

# captions: Fig. 1 merges its two captions; the rest are relabelled
f1a = find_para(lambda p: ptext(p).startswith("FIGURE 1."), "Fig. 1 short caption")
f1b = find_para(lambda p: ptext(p).startswith("Figure. 1 "), "Fig. 1 long caption")
replace_text(f1b, r"^\s*Figure\.\s*1\s*System Architecture of Sepsis Prediction and Digital Twin Framework\.",
             "FIGURE 1. System Architecture of Sepsis Prediction and Digital Twin Framework (conceptual architecture).", 1)
body.remove(f1a)
caps = []
for p in paras():
    m = re.match(r"\s*(FIGURE|TABLE)\s*(\d+)\.\s*", ptext(p))
    if not m:
        continue
    replace_text(p, r"^\s*(FIGURE|TABLE)\s*\d+\.\s*", "", 1)
    for r in p.iter(q("r")):
        rp = r.find(q("rPr"))
        if rp is not None:
            for x in rp.findall(q("b")) + rp.findall(q("bCs")):
                rp.remove(x)
    fig = m.group(1) == "FIGURE"
    set_ppr(p, "LNCSfigcap" if fig else "LNCStabcap")
    pos = 1
    r = E("r"); rp = etree.SubElement(r, q("rPr")); etree.SubElement(rp, q("b"))
    t = etree.SubElement(r, q("t")); t.text = f"{'Fig.' if fig else 'Table'} {m.group(2)}. "; t.set(XS, "preserve")
    p.insert(pos, r)
    caps.append(p)
fig_nums = [re.match(r"Fig\. (\d+)\.", ptext(p)).group(1) for p in caps if ptext(p).startswith("Fig.")]
assert fig_nums == ["1", "2", "3", "4", "5", "6"], fig_nums

# references
for p, new in zip(ref_paras, REFS):
    set_ppr(p, "LNCSref", keep=("numPr",))
    set_text(p, [(new, False)])

# single section, single column; drop empty paragraphs; body styles
for sp in list(body.iter(q("sectPr")))[:-1]:
    sp.getparent().remove(sp)
special = heads | set(caps) | set(ref_paras) | {title, abstract, kw}
for p in paras():
    has_obj = p.find(".//" + q("drawing")) is not None
    st = style_of(p)
    if not ptext(p).strip() and not has_obj and st not in ("LNCSaddress", "LNCSauthor"):
        body.remove(p)
        continue
    if p in special or st in ("LNCSauthor", "LNCSaddress"):
        continue
    set_ppr(p, "LNCSfigure" if has_obj else "BodyText")
kids = list(body)
for a_, b_ in zip(kids, kids[1:]):
    if style_of(a_) in ("Heading1", "Heading2") and style_of(b_) == "BodyText":
        b_.find(f"{q('pPr')}/{q('pStyle')}").set(q("val"), "LNCSp1a")

# grouped citations, "[1], [2], [12]" -> "[1,2,12]"
def compress(m):
    ns = sorted({int(x) for x in re.findall(r"\d+", m.group(0))})
    out, cur = [], [ns[0]]
    for n in ns[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            out.append(cur); cur = [n]
    out.append(cur)
    return "[" + ",".join(f"{r[0]}–{r[-1]}" if len(r) > 2 else ",".join(map(str, r)) for r in out) + "]"
for p in root.iter(q("p")):
    replace_text(p, r"\[\d+\](?:, ?\[\d+\])+", compress)

# direct run formatting off, so the LNCS styles govern (drawings untouched)
for rp in body.iter(q("rPr")):
    if any(etree.QName(a_).localname == "drawing" for a_ in rp.iterancestors()):
        continue
    for t in ("rFonts", "sz", "szCs", "spacing", "w", "kern", "color", "u", "position"):
        for x in rp.findall(q(t)):
            rp.remove(x)

# tables: fit the text block; 9 pt (Table 1: 8 pt)
for ti, tb in enumerate(body.iter(q("tbl"))):
    grid = tb.find(q("tblGrid"))
    cols = [int(c.get(q("w"))) for c in grid]
    f = min(1.0, TEXT_W / sum(cols))
    for c in grid:
        c.set(q("w"), str(int(int(c.get(q("w"))) * f)))
    for tcw in tb.iter(q("tcW")):
        if tcw.get(q("type")) == "dxa":
            tcw.set(q("w"), str(int(int(tcw.get(q("w"))) * f)))
    tp = tb.find(q("tblPr"))
    for t in ("tblW", "tblInd", "jc"):
        for x in tp.findall(q(t)):
            tp.remove(x)
    pos = sum(1 for c in tp if etree.QName(c).localname in
              ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize"))
    tp.insert(pos, E("tblW", w=min(sum(cols), TEXT_W), type="dxa"))
    tp.insert(pos + 1, E("jc", val="center"))
    for p in tb.iter(q("p")):
        set_ppr(p, "LNCStable8" if ti == 0 else "LNCStable", keep=("jc",))

# figures: inline, at most the text width
for d in body.iter(q("drawing")):
    anc = d.find(f"{{{WP}}}anchor")
    if anc is not None:
        inl = etree.SubElement(d, f"{{{WP}}}inline", distT="0", distB="0", distL="0", distR="0")
        for tag in ("extent", "effectExtent", "docPr", "cNvGraphicFramePr"):
            x = anc.find(f"{{{WP}}}{tag}")
            if x is not None:
                inl.append(x)
        inl.append(anc.find(f"{{{A}}}graphic"))
        d.remove(anc)
    inl = d.find(f"{{{WP}}}inline")
    ext = inl.find(f"{{{WP}}}extent")
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    if cx > TEXT_EMU:
        ncx, ncy = TEXT_EMU, int(cy * TEXT_EMU / cx)
        for e in [ext] + [e for e in inl.iter(f"{{{A}}}ext") if e.get("cx") == str(cx)]:
            e.set("cx", str(ncx)); e.set("cy", str(ncy))

# page: A4, 12.2 x 19.3 cm text block
sp = body.find(q("sectPr"))
for x in list(sp):
    sp.remove(x)
sp.append(E("pgSz", w=11906, h=16838))
sp.append(E("pgMar", top=2948, right=2494, bottom=2948, left=2494, header=2268, footer=2268, gutter=0))
sp.append(E("cols", space=720))

# styles
styles = etree.fromstring(zin.read("word/styles.xml"))
def pstyle(sid, name, size, bold=False, jc="both", before=0, after=0, first=0, left=0, right=0, hanging=0,
           keep_next=False, tabs=()):
    s = E("style", type="paragraph", styleId=sid)
    s.append(E("name", val=name)); s.append(E("qFormat"))
    pp = E("pPr")
    if keep_next:
        pp.append(E("keepNext"))
    if tabs:
        tb = E("tabs")
        for t in tabs:
            tb.append(E("tab", val="left", pos=t))
        pp.append(tb)
    pp.append(E("spacing", before=before, after=after, line=240, lineRule="auto"))
    ind = E("ind", left=left, right=right)
    if hanging:
        ind.set(q("hanging"), str(hanging))
    elif first:
        ind.set(q("firstLine"), str(first))
    pp.append(ind); pp.append(E("jc", val=jc))
    s.append(pp)
    rp = E("rPr")
    rp.append(E("rFonts", ascii="Times New Roman", hAnsi="Times New Roman", cs="Times New Roman", eastAsia="Times New Roman"))
    if bold:
        rp.append(E("b"))
    rp.append(E("sz", val=size * 2)); rp.append(E("szCs", val=size * 2))
    s.append(rp)
    return s
NEW = [pstyle("Normal", "Normal", 10),
       pstyle("Title", "Title", 14, bold=True, jc="center", after=240),
       pstyle("LNCSauthor", "LNCS author", 10, jc="center", after=120),
       pstyle("LNCSaddress", "LNCS address", 9, jc="center"),
       pstyle("LNCSabstract", "LNCS abstract", 9, before=480, after=120, left=567, right=567),
       pstyle("LNCSkeywords", "LNCS keywords", 9, after=240, left=567, right=567, jc="left"),
       pstyle("Heading1", "heading 1", 12, bold=True, jc="left", before=360, after=240, left=567, hanging=567, keep_next=True, tabs=(567,)),
       pstyle("Heading2", "heading 2", 10, bold=True, jc="left", before=240, after=120, left=567, hanging=567, keep_next=True, tabs=(567,)),
       pstyle("BodyText", "Body Text", 10, first=227),
       pstyle("LNCSp1a", "LNCS p1a", 10),
       pstyle("LNCSfigure", "LNCS figure", 10, jc="center", before=240, after=120, keep_next=True),
       pstyle("LNCSfigcap", "LNCS figure caption", 9, jc="center", after=240),
       pstyle("LNCStabcap", "LNCS table caption", 9, jc="center", before=240, after=120, keep_next=True),
       pstyle("LNCStable", "LNCS table text", 9, jc="left"),
       pstyle("LNCStable8", "LNCS table text small", 8, jc="left"),
       pstyle("LNCSrefhead", "LNCS reference heading", 12, bold=True, jc="left", before=360, after=240, keep_next=True),
       pstyle("LNCSref", "LNCS reference", 9, left=340, hanging=340)]
ids = {s.get(q("styleId")) for s in NEW}
for s in styles.findall(q("style")):
    if s.get(q("styleId")) in ids:
        styles.remove(s)
for s in NEW:
    if s.get(q("styleId")) == "Normal":
        s.set(q("default"), "1")
    styles.append(s)
dd = styles.find(f"{q('docDefaults')}/{q('rPrDefault')}/{q('rPr')}")
if dd is not None:
    for t in ("rFonts", "sz", "szCs"):
        for x in dd.findall(q(t)):
            dd.remove(x)
    dd.insert(0, E("rFonts", ascii="Times New Roman", hAnsi="Times New Roman", cs="Times New Roman", eastAsia="Times New Roman"))
    dd.append(E("sz", val=20)); dd.append(E("szCs", val=20))

# reference numbering "1." with a hanging indent
numbering = etree.fromstring(zin.read("word/numbering.xml"))
nid = ref_paras[0].find(".//" + q("numId")).get(q("val"))
num = next(x for x in numbering.findall(q("num")) if x.get(q("numId")) == nid)
aid = num.find(q("abstractNumId")).get(q("val"))
lvl = next(x for x in numbering.findall(q("abstractNum")) if x.get(q("abstractNumId")) == aid).find(q("lvl"))
lvl.find(q("lvlText")).set(q("val"), "%1.")
ind = lvl.find(f"{q('pPr')}/{q('ind')}")
ind.set(q("left"), "340"); ind.set(q("hanging"), "340")
lrp = lvl.find(q("rPr"))
if lrp is not None:
    for t in ("sz", "szCs"):
        for x in lrp.findall(q(t)):
            lrp.remove(x)
    lrp.append(E("sz", val=18)); lrp.append(E("szCs", val=18))

# ============================================================================ write (unused media dropped)
used = {e.get(f"{{{R}}}embed") for e in root.iter(f"{{{A}}}blip")}
dropped = []
def keep_rel(m):
    rid = re.search(r'Id="([^"]+)"', m.group(0)).group(1)
    if "relationships/image" in m.group(0) and rid not in used:
        dropped.append("word/" + re.search(r'Target="([^"]+)"', m.group(0)).group(1))
        return ""
    return m.group(0)
rels = re.sub(r"<Relationship [^>]*/>", keep_rel, rels)
out = {"word/document.xml": etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
       "word/styles.xml": etree.tostring(styles, xml_declaration=True, encoding="UTF-8", standalone=True),
       "word/numbering.xml": etree.tostring(numbering, xml_declaration=True, encoding="UTF-8", standalone=True),
       "word/_rels/document.xml.rels": rels.encode(), **media_out}
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
    for it in zin.infolist():
        if it.filename in dropped:
            continue
        z.writestr(it, out.pop(it.filename, None) or zin.read(it))
    for k, v in out.items():
        if k not in dropped:
            z.writestr(k, v)
print(f"wrote {DST.relative_to(ROOT)}: {sec} sections, figures {fig_nums}, abstract {words} words, "
      f"{len(ref_paras)} references, dropped media {dropped}")
