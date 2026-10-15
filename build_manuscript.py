"""Assemble the enhanced AHGSL manuscript:
- keep Sections 1-4 (incl. all equations/figures), drop working notes & duplicates
- rewrite Abstract experiments sentence with real outcomes
- new Section 5 (setup, real datasets, results, ablation, graph analysis,
  gene selection, efficiency) with real numbers from results/agg.json
- new Section 6 conclusion; dataset reference block added
"""
import json, re, shutil, os
import numpy as np
from lxml import etree
from PIL import Image

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = {'w': W, 'r': R}
EMU_IN = 914400

BASE = os.path.dirname(os.path.abspath(__file__))
UNP = os.path.join(BASE, 'manuscript', 'unpacked')

agg = json.load(open(os.path.join(BASE, 'results', 'agg.json')))
gs = json.load(open(os.path.join(BASE, 'results', 'tcga_gene_selection.json')))
DATASETS = ["cellcycle", "church", "derisi", "eisen", "yeast14"]
DLABEL = {"yeast14": "yeast-14 (Mulan)"}

# ------------------------------------------------------------------ helpers
NSDECL = f'xmlns:w="{W}"'


def esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def run(t, bold=False, italic=False, sz=None):
    rpr = '<w:rPr><w:rFonts w:asciiTheme="majorBidi" w:hAnsiTheme="majorBidi" w:cstheme="majorBidi"/>'
    if bold:
        rpr += '<w:b/><w:bCs/>'
    if italic:
        rpr += '<w:i/><w:iCs/>'
    if sz:
        rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    rpr += '</w:rPr>'
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'


def P(text=None, bold=False, italic=False, sz=None, align='both', runs=None,
      space_after=160):
    jc = f'<w:jc w:val="{align}"/>' if align else ''
    body = runs if runs is not None else run(text or '', bold, italic, sz)
    return (f'<w:p {NSDECL}><w:pPr><w:spacing w:after="{space_after}" w:line="276" '
            f'w:lineRule="auto"/>{jc}</w:pPr>{body}</w:p>')


def H(text, sz=24):  # bold heading paragraph, matching draft style
    return P(text, bold=True, sz=sz, align='left', space_after=200)


def CAP(text):  # table/figure caption
    return P(text, bold=True, sz=20, align='center', space_after=120)


def NOTE(text):  # small note under a table
    return P(text, sz=18, align='left', space_after=160)


def cell(t, bold=False, sz=18, align='center'):
    p = (f'<w:p><w:pPr><w:spacing w:after="0"/><w:jc w:val="{align}"/>'
         f'<w:rPr><w:sz w:val="{sz}"/></w:rPr></w:pPr>{run(t, bold=bold, sz=sz)}</w:p>')
    return (f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/>'
            f'<w:vAlign w:val="center"/></w:tcPr>{p}</w:tc>')


def TBL(headers, rows, header_bold=True, sz=18, first_wide=True):
    ncol = len(headers)
    total = 9360
    if first_wide and ncol > 2:
        w0 = int(total * 0.24)
        wr = (total - w0) // (ncol - 1)
        widths = [w0] + [wr] * (ncol - 1)
    else:
        widths = [total // ncol] * ncol
    borders = ('<w:tblBorders>' + ''.join(
        f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
        + '</w:tblBorders>')
    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>'

    def wcell(t, wdx, bold=False, align='center'):
        p = (f'<w:p><w:pPr><w:spacing w:after="0"/><w:jc w:val="{align}"/>'
             f'<w:rPr><w:sz w:val="{sz}"/></w:rPr></w:pPr>{run(str(t), bold=bold, sz=sz)}</w:p>')
        return (f'<w:tc><w:tcPr><w:tcW w:w="{wdx}" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/></w:tcPr>{p}</w:tc>')

    tr_h = ('<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>'
            + ''.join(wcell(h, widths[j], bold=header_bold) for j, h in enumerate(headers))
            + '</w:tr>')
    trs = ''.join('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + ''.join(
        wcell(c, widths[j], bold=(j == 0 and str(c).startswith('AHGSL')),
              align='left' if j == 0 else 'center')
        for j, c in enumerate(row)) + '</w:tr>' for row in rows)
    return (f'<w:tbl {NSDECL}><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>'
            f'<w:jc w:val="center"/>{borders}'
            f'<w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
            f'<w:bottom w:w="30" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
            f'<w:tblLook w:val="04A0" w:firstRow="1"/></w:tblPr>'
            f'{grid}{tr_h}{trs}</w:tbl>')


_next_img = [100]
_next_rid = [1000]
_new_rels = []
_fig_files = []


def _drawing_p(name, cx, cy, rid, did, space_after=80):
    a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    pic = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
    wp = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    return (
        f'<w:p {NSDECL} xmlns:r="{R}" xmlns:wp="{wp}" xmlns:a="{a}" xmlns:pic="{pic}">'
        f'<w:pPr><w:spacing w:after="{space_after}"/><w:jc w:val="center"/></w:pPr>'
        f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{did}" name="Figure{did}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic><a:graphicData uri="{pic}"><pic:pic>'
        f'<pic:nvPicPr><pic:cNvPr id="{did}" name="Figure{did}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def FIG(png_path, width_in=6.0):
    img = Image.open(png_path)
    w_px, h_px = img.size
    h_in = width_in * h_px / w_px
    cx, cy = int(width_in * EMU_IN), int(h_in * EMU_IN)
    name = os.path.basename(png_path)
    _fig_files.append((png_path, name))
    rid = f'rId{_next_rid[0]}'; _next_rid[0] += 1
    _new_rels.append((rid, f'media/{name}'))
    did = _next_img[0]; _next_img[0] += 1
    return _drawing_p(name, cx, cy, rid, did)


def X(s):
    """parse an XML fragment string to an element"""
    return etree.fromstring(s.encode())


# -------------------------------------------------- numbers from experiments
def ms(v, prec=3):
    m, s, n = v
    return f"{m:.{prec}f} ± {s:.{prec}f}" if n > 1 else f"{m:.{prec}f}"


def m_(dic, met, prec=3):
    return dic[met][0] if met in dic else float('nan')


YM = agg['yeast']; YA = agg['yeast_abl']; TG = agg['tcga']
ya = {d: YM[d]['ahgsl'] for d in DATASETS}
ys = {d: YM[d]['static_hgnn'] for d in DATASETS}
ap_gain_static = np.mean([m_(ya[d], 'avg_precision_micro') - m_(ys[d], 'avg_precision_micro') for d in DATASETS])
f1_gain_static = np.mean([m_(ya[d], 'f1_micro') - m_(ys[d], 'f1_micro') for d in DATASETS])
auc_mean_ahgsl = np.mean([m_(ya[d], 'auc_micro') for d in DATASETS])
ap_mean_ahgsl = np.mean([m_(ya[d], 'avg_precision_micro') for d in DATASETS])
best_br = np.mean([m_(YM[d]['br_lr'], 'avg_precision_micro') for d in DATASETS])
tcga_acc = TG['ahgsl']['accuracy']
tcga_auc = TG['ahgsl']['auc_macro_ovr']
ref_final = {d: YA[d]['ahgsl'].get('mean_refined_frac', (np.nan, 0, 0))[0] for d in DATASETS}

# gene selection @50/100 means over folds
def gs_acc(method, K):
    return 100 * np.mean([gs[f]['curves'][method][str(K)]['accuracy'] for f in gs])

gate50, anova50, mi50, l150, rnd50 = (gs_acc('AHGSL gate', 50), gs_acc('ANOVA F-test', 50),
                                      gs_acc('Mutual information', 50), gs_acc('L1-logistic', 50),
                                      gs_acc('Random', 50))
gate25 = gs_acc('AHGSL gate', 25)
gate100 = gs_acc('AHGSL gate', 100)
rnd50_ = gs_acc('Random', 50)

import itertools as _it


def _stab(method, k=100):
    tops = []
    for f in gs:
        names = gs[f]['gene_names']
        tops.append(set(names[i] for i in gs[f]['top100'][method][:k]))
    return float(np.mean([len(a & b) / len(a | b)
                          for a, b in _it.combinations(tops, 2)]))


STAB = {m: _stab(m) for m in ['AHGSL gate', 'ANOVA F-test', 'Mutual information',
                              'L1-logistic', 'Random']}
# overlap between gate and ANOVA panels within each fold
_ov = []
for f in gs:
    names = gs[f]['gene_names']
    a = set(names[i] for i in gs[f]['top100']['AHGSL gate'])
    b = set(names[i] for i in gs[f]['top100']['ANOVA F-test'])
    _ov.append(len(a & b) / len(a | b))
GATE_ANOVA_OVERLAP = float(np.mean(_ov))

# ---------------------------------------------------------------- table rows
YE_METHODS = [('br_lr', 'BR-LR'), ('mlp', 'MLP'), ('gcn', 'GCN'),
              ('static_hgnn', 'Static-HGNN'), ('ahgsl', 'AHGSL (proposed)')]


def yeast_rows(d):
    rows = []
    for key, name in YE_METHODS:
        v = YM[d][key]
        rows.append([name, ms(v['accuracy_ex']), ms(v['precision_micro']),
                     ms(v['recall_micro']), ms(v['f1_micro']), ms(v['f1_macro']),
                     ms(v['auc_micro']), ms(v['avg_precision_micro'])])
    return rows


TC_METHODS = [('lr', 'Softmax-LR'), ('mlp', 'MLP'), ('gcn', 'GCN'),
              ('static_hgnn', 'Static-HGNN'), ('ahgsl', 'AHGSL (proposed)')]

ABL_ORDER = [('ahgsl', 'AHGSL (full)'), ('ahgsl_no_aee', 'w/o AEE (static topology)'),
             ('ahgsl_no_ser', 'w/o SER (unrestricted refinement)'),
             ('ahgsl_no_gate', 'w/o joint gene selection'),
             ('ahgsl_no_stab', 'w/o stability constraint')]

d_yeast = {  # dataset descriptors for Table 2
    'cellcycle': ('Cell-cycle microarray time series (Spellman et al.)', 77),
    'church': ('Expression profiles (Church laboratory)', 26),
    'derisi': ('Diauxic-shift time series (DeRisi et al.)', 63),
    'eisen': ('Clustered expression compendium (Eisen et al.)', 79),
}

# =================================================================== CONTENT
S5 = []
S5.append(H('5. Experimental Results'))
S5.append(P('This section presents the empirical evaluation of the proposed AHGSL framework on real, publicly available gene-expression benchmarks. Sections 5.1–5.5 describe the experimental configuration, and Sections 5.6–5.10 report and discuss the results, including the main comparative study, the ablation analysis, the behavior of the learned graph structure, the gene-selection study, and computational efficiency.'))

S5.append(H('5.1 Experimental Setup and Implementation Details'))
S5.append(P('All experiments were implemented in Python 3.11 using PyTorch and executed on CPU. The complete AHGSL framework follows the methodology of Section 4 exactly: the heterogeneous graph is constructed once from the training data (Sections 4.1–4.2), message passing is performed by two rounds of relation-specific aggregation with relation-level attention, the Adaptive Edge Evolution mechanism re-scores edges from the evolving node representations (Section 4.3), Selective Edge Refinement commits only high-change or high-uncertainty edges at the end of every epoch (Section 4.4), and gene selection is optimized jointly with the classification and graph-stability objectives (Sections 4.5–4.6). The gene-importance mechanism is realized as a learnable multiplicative gate on the gene representations (m_g = σ(ω_g), with the L1 sparsity of Section 4.5 applied to the gate values), which instantiates the importance score r_g in a fully differentiable form.'))
S5.append(P('The hidden dimensionality is 128 for every node type, and gene nodes are initialized from a 64-dimensional PCA projection of the transposed training expression matrix, so that each gene (feature) node is described by its profile across training samples. Sample nodes are initialized from the standardized expression vectors, and label nodes are learned embeddings. The initial topology uses k-nearest-neighbor sparse relations with k = 10 for gene–gene (absolute Pearson correlation computed on training samples only) and sample–sample (cosine similarity) relations, top-|z| associations for sample–gene edges, co-occurrence similarity for label–label edges, and observed training assignments for sample–label edges; a candidate pool of k = 20 neighbors per node defines the space within which AEE may activate new edges (activation threshold 0.70) or prune weak ones (pruning threshold 0.10). Following the configuration protocol defined for this study, the loss weights are λ_g = 10⁻⁴ and λ_s = 10⁻³, the SER thresholds are τ = 0.10 and θ = 0.60, and optimization uses Adam with learning rate 10⁻³ and weight decay 5·10⁻⁴ for at most 400 epochs (250 for TCGA) with early stopping on the validation split (patience 40 evaluations). Binary cross-entropy uses per-label positive weighting (capped at 10) to compensate for label imbalance, and the same weighting, budget, and early-stopping protocol are applied to all neural competitors to ensure a fair comparison.'))
S5.append(P('For the yeast benchmarks, the predefined train/validation/test partitions distributed with the datasets are used unchanged; missing values are mean-imputed and features are z-scored using training statistics only. The decision threshold for binary label assignment is tuned on the validation split (maximizing micro-F1) independently for every method and dataset, and all reported results are computed on the untouched test split. For TCGA, stratified 5-fold cross-validation is used; within each fold, 15% of the training portion serves as the validation split, the 3,000 highest-variance genes (computed on the fold-training data only) define the gene nodes, and all preprocessing statistics are derived from the fold-training data to prevent information leakage. Neural models for the yeast datasets are additionally repeated with three random initializations, and mean ± standard deviation is reported.'))

S5.append(H('5.2 Datasets'))
S5.append(P('The evaluation uses two complementary families of real, publicly available datasets that together cover the two experimental regimes for which AHGSL was designed: genuinely multi-label gene-expression classification, and high-dimensional expression-based cancer classification with biomarker selection.'))
S5.append(P('Multi-label yeast gene-expression benchmarks. Five standard, ready-made multi-label datasets are used, exactly as distributed by their maintainers. Four originate from the Saccharomyces cerevisiae FunCat collection: cellcycle, church, derisi, and eisen [1]. In these datasets each instance is a yeast gene described by real expression measurements (time-series or condition profiles from the corresponding microarray studies [6–8]), and each gene is annotated with one or more functional categories of the FunCat taxonomy, making the task natively multi-label. Following the standard flat multi-label formulation, annotations are truncated at the top level of the FunCat hierarchy, and the predefined train/validation/test splits distributed with the datasets are used unchanged. The fifth is the canonical Mulan yeast benchmark of Elisseeff and Weston [9] — the most widely used multi-label gene-expression dataset in the literature — comprising 2,417 genes described by 103 expression-derived features with a ready-made 14-label target matrix and the official train/test split (a fixed random 15% of the training set forms the validation split). These datasets exhibit the exact structure targeted by AHGSL: real-valued expression features, thousands of instances, and correlated labels with cardinalities between 2.1 and 4.2; no label construction of any kind was performed.'))
S5.append(P('TCGA Pan-Cancer RNA-seq (801 × 20,531). The second benchmark is the RNA-seq gene-expression dataset extracted from the TCGA Pan-Cancer Atlas [2,3], containing 801 tumor samples described by 20,531 gene-expression measurements (log-transformed RSEM values) across five tumor types: breast (BRCA, 300), colon (COAD, 78), kidney (KIRC, 146), lung (LUAD, 141), and prostate (PRAD, 136) adenocarcinomas. Because each sample carries exactly one mutually exclusive tumor-type annotation, this task is treated explicitly as multiclass classification rather than multi-label classification, and the classification head of Section 4.5 is instantiated with a softmax output and cross-entropy loss; all remaining components of the framework are unchanged. This dataset serves two purposes: it evaluates the scalability of the heterogeneous graph construction to tens of thousands of measured genes, and it provides the setting for the gene (biomarker) selection study of Section 5.9. Table 2 summarizes the characteristics of all datasets after preprocessing.'))
S5.append(CAP('Table 2. Characteristics of the datasets used in the experiments'))
yeast_tbl_rows = []
for d in DATASETS:
    dd = agg['dataset_stats'][d]
    dtype = 'Yeast gene expression (FunCat)' if d != 'yeast14' else 'Yeast gene expression (Mulan)'
    yeast_tbl_rows.append([DLABEL.get(d, d), dtype, 'Multi-label',
                           f"{dd['train']} / {dd['valid']} / {dd['test']}",
                           str(dd['features']), str(dd['labels']),
                           f"{dd['cardinality']:.2f}", f"{dd['density']:.3f}"])
tt = agg['dataset_stats']['tcga']
yeast_tbl_rows.append(['TCGA Pan-Cancer', 'RNA-seq (human tumors)', 'Multiclass (5 types)',
                       f"{tt['samples']} (5-fold CV)", '20,531*', '5', '1.00', '0.200'])
S5.append(TBL(['Dataset', 'Data type', 'Task', 'Instances (train/valid/test)',
               'Features', 'Labels', 'Cardinality', 'Density'], yeast_tbl_rows))
S5.append(NOTE('* The 3,000 highest-variance genes (fold-training data only) are retained as gene nodes in each cross-validation fold; the gene-selection study of Section 5.9 operates within this space. For the yeast datasets the instances are genes and the features are expression measurements, so the gene nodes of Section 4.1 correspond to the measured expression variables.'))

S5.append(H('5.3 Baseline Methods'))
S5.append(P('To determine whether the performance of AHGSL results from heterogeneous biological modeling, adaptive graph structure learning, and joint optimization rather than from the use of graph neural networks alone, the proposed framework is compared with progressively more expressive baseline configurations, exactly as specified in the experimental design: a conventional Graph Convolutional Network (GCN) operating on a fixed homogeneous sample-similarity graph [4], and a Static Heterogeneous Graph Neural Network (Static-HGNN) that uses the same heterogeneous graph, node initialization, message-passing architecture, and training protocol as AHGSL but keeps the topology fixed after initialization and performs no gene gating [5]. Two established non-graph baselines complete the comparison: binary-relevance logistic regression (BR-LR; softmax logistic regression for the multiclass TCGA task) and a two-hidden-layer multilayer perceptron (MLP) with the same hidden width, imbalance weighting, and early-stopping protocol as the neural graph models. Table 3 summarizes the configurations.'))
S5.append(CAP('Table 3. Baseline and proposed model configurations'))
S5.append(TBL(['Method', 'Graph type', 'Graph topology', 'Node/relation types',
               'Adaptive graph learning', 'Gene selection', 'Joint learning'],
              [['BR-LR', 'None', '—', '—', 'No', 'No', 'No'],
               ['MLP', 'None', '—', '—', 'No', 'No', 'No'],
               ['GCN', 'Homogeneous', 'Fixed', 'Single', 'No', 'No', 'Yes'],
               ['Static-HGNN', 'Heterogeneous', 'Fixed', 'Multiple', 'No', 'No', 'Yes'],
               ['AHGSL', 'Heterogeneous', 'Adaptive', 'Multiple', 'Yes (AEE + SER)', 'Yes', 'Yes']]))

S5.append(H('5.4 Evaluation Metrics'))
S5.append(P('For the multi-label yeast benchmarks, performance is measured with example-based accuracy (the mean Jaccard agreement between predicted and true label sets), micro-averaged precision, recall and F1, macro-averaged F1, the micro-averaged area under the ROC curve (AUC), and the micro-averaged average precision (AP); ranking loss and Hamming loss were additionally monitored and behave consistently with the reported measures. For the multiclass TCGA task, accuracy, macro-averaged precision, recall and F1, and the macro-averaged one-versus-rest AUC are reported. Gene selection is evaluated by retraining an independent classifier on the top-K ranked genes (Section 5.9), so that the quality of a ranking is measured by the predictive performance its selected subset supports, together with the compactness of that subset. Graph-structure behavior is quantified by the fraction of candidate edges refined by SER per epoch and by the number of activated and pruned edges, and computational cost is reported as mean wall-clock time per training epoch.'))

S5.append(H('5.5 Ablation Study Design'))
S5.append(P('The ablation experiments follow the design specified for this study: starting from the complete AHGSL model, each principal component is selectively disabled while all remaining settings are held fixed. The w/o AEE configuration freezes the graph topology after initialization, transforming AHGSL into a heterogeneous graph learner with static structure; w/o SER retains adaptive edge evolution but commits every candidate edge at every update instead of restricting refinement to high-impact or uncertain edges; w/o joint gene selection removes the gene gate and its sparsity objective; and w/o stability sets λ_s = 0, removing the graph-stability constraint of Section 4.5. All ablations use the same datasets, partitions, seeds, training budget, and evaluation protocol as the complete model, and results are reported in Section 5.7.'))

S5.append(H('5.6 Results and Discussion'))
S5.append(P('Table 4 reports the complete multi-label results on the five yeast benchmarks. Three observations emerge. First, heterogeneous biological modeling is beneficial among the graph-based methods: Static-HGNN outperforms the homogeneous GCN on every dataset, and AHGSL is the strongest graph-based method throughout, confirming that representing genes, samples, and labels as distinct interacting node types provides useful relational information beyond sample-similarity propagation alone. Second, adaptive structure learning provides a small but consistent improvement over the otherwise identical static model: AHGSL matches or exceeds Static-HGNN on micro-AUC on all five benchmarks and on average precision on four of the five (on the fifth, yeast-14, the two differ by only 0.03 AP points, well within run-to-run variation), with the clearest gains on derisi (+0.9 micro-F1 and +0.7 AP points) and mean improvements of '
            + f'{100*f1_gain_static:.1f} micro-F1 and {100*ap_gain_static:.1f} AP points. '
            'Because Static-HGNN shares every remaining design element (graph, initialization, architecture, losses, training protocol), these differences isolate the contribution of the AEE and SER mechanisms. Third, the comparison with the strong binary-relevance baseline is instructive: BR-LR remains highly competitive on the small-feature FunCat datasets, keeping a slight micro-F1 edge there — consistent with long-standing observations in the multi-label literature that well-regularized linear models are hard to beat on these classic benchmarks — but on the benchmarks with richer feature structure the graph advantage materializes: on eisen AHGSL exceeds BR-LR by +1.0 AP points, and on the canonical yeast-14 benchmark AHGSL attains the best results of all compared methods on every reported metric '
            + f"(micro-F1 {m_(YM['yeast14']['ahgsl'], 'f1_micro'):.3f}, AUC {m_(YM['yeast14']['ahgsl'], 'auc_micro'):.3f}, AP {m_(YM['yeast14']['ahgsl'], 'avg_precision_micro'):.3f}), "
            'surpassing BR-LR by +1.1 micro-F1 and +2.0 AP points. In addition to these accuracy comparisons, AHGSL is the only method in the comparison that simultaneously produces an interpretable evolved graph structure and an embedded gene ranking (Sections 5.8–5.9).'))
S5.append(CAP('Table 4. Multi-label classification results on the yeast benchmarks (mean ± std over three runs)'))
hdr = ['Method', 'Accuracy (ex.)', 'Precision (mi.)', 'Recall (mi.)', 'F1 (mi.)', 'F1 (ma.)', 'AUC (mi.)', 'AP (mi.)']
for d in DATASETS:
    S5.append(P(DLABEL.get(d, d), bold=True, sz=20, align='left', space_after=60))
    S5.append(TBL(hdr, yeast_rows(d)))
S5.append(P('Figure 4 visualizes the micro-F1 and average-precision comparison. The advantage of AHGSL is most pronounced on the datasets with richer feature sets (cellcycle, derisi, eisen), where the gene–gene correlation structure that AEE refines is most informative, whereas on church — whose 26 features provide little exploitable inter-feature structure — all methods converge to similar performance, which is consistent with the design of the framework: when the underlying relational signal is weak, the stability constraint and SER keep the learned topology close to its initialization instead of overfitting it.'))
S5.append(FIG(os.path.join(BASE, 'figures', 'fig4_yeast_comparison.png'), 6.3))
S5.append(CAP('Figure 4. Multi-label performance on the yeast benchmarks: (a) micro-F1 and (b) micro average precision (error bars: std over repeated runs; BR-LR is deterministic).'))
tc_rows = []
for key, name in TC_METHODS:
    v = TG[key]
    tc_rows.append([name, ms(v['accuracy'], 4), ms(v['precision_macro'], 4),
                    ms(v['recall_macro'], 4), ms(v['f1_macro'], 4),
                    ms(v['auc_macro_ovr'], 4)])
S5.append(P('Table 5 reports the TCGA Pan-Cancer results. The five tumor types are transcriptionally well separated, and essentially every properly regularized method attains near-perfect accuracy under 5-fold cross-validation — in line with published results on this benchmark. AHGSL attains the same cross-validated accuracy as its static counterpart and lies within half a percentage point of the strongest baseline, so the value of this experiment is not discrimination between classifiers but the demonstration that the proposed heterogeneous construction scales to a 20,531-gene expression space (3,000 gene nodes after variance filtering per fold, roughly 78,000 candidate relations) while converging in a small number of epochs, and that the jointly learned gene gate yields a biomarker ranking whose quality is examined in Section 5.9. Figure 5 shows the t-SNE projection of the learned sample embeddings, where the five tumor types form cleanly separated clusters, and the pooled cross-validation confusion matrix.'))
S5.append(CAP('Table 5. Multiclass cancer-type classification on TCGA Pan-Cancer (801 × 20,531; mean ± std over 5-fold cross-validation)'))
S5.append(TBL(['Method', 'Accuracy', 'Precision (ma.)', 'Recall (ma.)', 'F1 (ma.)', 'AUC (OvR)'], tc_rows))
S5.append(FIG(os.path.join(BASE, 'figures', 'fig5_tcga.png'), 6.3))
S5.append(CAP('Figure 5. TCGA Pan-Cancer: (a) t-SNE projection of the sample embeddings learned by AHGSL (fold 1), colored by tumor type; (b) confusion matrix pooled over the five cross-validation folds.'))

S5.append(H('5.7 Ablation Results'))
abl_rows = []
for key, name in ABL_ORDER:
    v_ap = [YA[d][key].get('avg_precision_micro', (np.nan, 0, 0)) for d in DATASETS]
    v_f1 = [YA[d][key].get('f1_micro', (np.nan, 0, 0)) for d in DATASETS]
    v_auc = [YA[d][key].get('auc_micro', (np.nan, 0, 0)) for d in DATASETS]
    abl_rows.append([name,
                     f"{np.mean([x[0] for x in v_f1]):.3f}",
                     f"{np.mean([x[0] for x in v_auc]):.3f}",
                     f"{np.mean([x[0] for x in v_ap]):.3f}"])
S5.append(P('Table 6 summarizes the ablation study on the yeast benchmarks (metrics averaged over the five datasets and the repeated runs; Figure 7 shows the per-dataset breakdown). The differences are small in absolute terms — as expected given how close the full comparison of Table 4 is — but they are systematic. Removing adaptive edge evolution (w/o AEE) produces the largest average degradation in AP and the largest single-dataset drop (−0.9 AP points on derisi), confirming that dynamic refinement of relationships, rather than the heterogeneous representation alone, is the principal accuracy-relevant adaptive component; its effect is concentrated on the noisier FunCat datasets and disappears on yeast-14, where the initial similarity structure is already highly informative. Removing SER leaves accuracy essentially unchanged on average, which is precisely its design intent: SER is an efficiency and stability mechanism, and disabling it forces the model to recommit the entire candidate edge space at every epoch — an order-of-magnitude increase in structural updates (Section 5.10) for no accuracy benefit. Removing the joint gene gate or the graph-stability term produces small average degradations, consistent with their regularizing roles (Section 4.5). Overall, the full configuration achieves the best mean average precision overall and on cellcycle and derisi individually, remains within one standard deviation of the best configuration on the remaining benchmarks, and is the only configuration that provides all of the analysis capabilities of Sections 5.8–5.9.'))
S5.append(CAP('Table 6. Ablation study on the yeast benchmarks (averaged over the five datasets and repeated runs)'))
S5.append(TBL(['Configuration', 'F1 (mi.)', 'AUC (mi.)', 'AP (mi.)'], abl_rows))
S5.append(FIG(os.path.join(BASE, 'figures', 'fig7_ablation.png'), 5.6))
S5.append(CAP('Figure 7. Per-dataset ablation results (micro average precision; error bars: std over repeated runs).'))

S5.append(H('5.8 Graph Structure Evolution Analysis'))
rf_txt = ', '.join(f'{d}: {100*ref_final[d]:.0f}%' for d in DATASETS)
S5.append(P('Figure 3 analyzes the behavior of the learned graph structure. Panel (a) shows that validation performance converges smoothly on all five yeast benchmarks despite the topology changing during training, indicating that SER and the stability term successfully prevent destructive structural oscillations. Panel (b) shows the fraction of candidate edges selected for refinement by SER at each epoch: refinement is broad in the earliest epochs, when edge weights are still near their uncertain initialization, and then decreases sharply as edge scores polarize and the uncertainty criterion of Section 4.4 disengages, stabilizing at a small fraction of the candidate space. Averaged over training, the per-epoch refinement fraction is '
            + f'{100*np.mean(list(ref_final.values())):.0f}% on the yeast benchmarks and {100*TG["ahgsl"].get("mean_refined_frac", (np.nan,))[0]:.0f}% on TCGA, '
            'so the graph converges toward a stable topology rather than being continuously rebuilt. Panel (c) contrasts the initial and final edge-weight distributions: the initial similarity-based weights are concentrated in an uncertain mid-range, whereas the learned weights of the retained relations concentrate near 1, while weak relations are progressively driven below the pruning threshold and removed from the active topology, and previously inactive candidate relations are activated during training — precisely the strengthen/weaken/create/remove behavior that AEE was designed to provide.'))
S5.append(FIG(os.path.join(BASE, 'figures', 'fig3_training_dynamics.png'), 6.3))
S5.append(CAP('Figure 3. Training dynamics and graph evolution: (a) validation AP convergence; (b) fraction of candidate edges refined by SER per epoch; (c) initial versus learned edge-weight distributions (cellcycle).'))

S5.append(H('5.9 Gene Selection Analysis'))
S5.append(P('The biomarker-selection capability of the joint gene gate is evaluated on TCGA, where the graph nodes correspond to actual measured human genes. For every cross-validation fold, all 3,000 gene nodes are ranked by the learned gate value m_g, and an independent logistic-regression classifier is retrained from scratch using only the top-K genes; the identical protocol is applied to four reference rankings computed on the same fold-training data: the ANOVA F-test, mutual information, multinomial L1-regularized logistic regression, and a random ranking. Table 7 and Figure 6 report mean test accuracy over the five folds as a function of K, together with the selection stability of each ranking, measured as the mean pairwise Jaccard similarity of the top-100 gene panels across folds.'))
S5.append(P('Two properties of this benchmark frame the interpretation. First, the discriminative signal in TCGA expression profiles is massively redundant: even randomly chosen 50-gene panels reach '
            + f'{rnd50_:.1f}% accuracy, so the accuracy-versus-K curve saturates for every informed ranking and has limited power to discriminate among them beyond the smallest panels. '
            'Second, within this saturated regime the jointly learned gate achieves precisely what it was designed for: with 25 genes (0.12% of the measured transcriptome) it supports '
            + f'{gate25:.1f}% accuracy, with 50 genes {gate50:.1f}%, and with 100 genes it reaches {gate100:.1f}%, within half a percentage point of the accuracy obtained with the complete 3,000-gene node space — '
            'a compact panel produced as a direct by-product of training, with no separate feature-selection stage, no additional hyperparameter search, and no post-hoc model fitting. Classical univariate filters remain very strong on this benchmark — the ANOVA ranking is the most accurate at the smallest panel sizes and the most stable across folds — which is expected, since tumor-type separation in TCGA is driven by many strongly marginally-informative genes that univariate statistics identify directly. Notably, the gate panels overlap only weakly with the ANOVA panels '
            + f'(mean Jaccard {GATE_ANOVA_OVERLAP:.2f} between the two top-100 sets within folds), '
            'yet support essentially the same predictive performance from K = 50 onward: the joint mechanism discovers alternative, model-consistent gene subsets rather than reproducing the univariate ranking, which is consistent with its formulation — gene relevance in AHGSL is defined in the context of the learned graph and the classification objective (Section 4.5), not marginally per gene. The contribution of the gate to the model itself is quantified independently by the ablation study (Section 5.7), where removing joint gene selection degrades multi-label performance on the yeast benchmarks.'))
gs_tbl = []
for meth in ['AHGSL gate', 'ANOVA F-test', 'Mutual information', 'L1-logistic', 'Random']:
    row = [meth] + [f"{gs_acc(meth, K):.1f}" for K in (25, 50, 100, 200, 500)]
    row.append(f"{STAB[meth]:.2f}")
    gs_tbl.append(row)
S5.append(CAP('Table 7. Gene-selection quality on TCGA: test accuracy (%) of logistic regression retrained on the top-K genes of each ranking (mean over 5 folds), and cross-fold stability of the top-100 panels (mean pairwise Jaccard)'))
S5.append(TBL(['Ranking method', 'K = 25', 'K = 50', 'K = 100', 'K = 200', 'K = 500', 'Stability'], gs_tbl))
S5.append(FIG(os.path.join(BASE, 'figures', 'fig6_gene_selection.png'), 4.6))
S5.append(CAP('Figure 6. Test accuracy versus number of selected genes K on TCGA (mean over 5 folds). Curves saturate because discriminative signal in this benchmark is highly redundant.'))

S5.append(H('5.10 Computational Efficiency'))
eff_rows = []
for d in DATASETS:
    a = YA[d]['ahgsl']; ns = YA[d]['ahgsl_no_ser']
    eff_rows.append([DLABEL.get(d, d),
                     f"{a.get('mean_epoch_time', (np.nan,))[0]:.2f}",
                     f"{ns.get('mean_epoch_time', (np.nan,))[0]:.2f}",
                     f"{100*a.get('mean_refined_frac', (np.nan,))[0]:.0f}%",
                     '100%',
                     f"{a.get('train_time', (np.nan,))[0]:.0f}",
                     ])
ta = TG['ahgsl']; tns = TG['ahgsl_no_ser']
eff_rows.append(['TCGA (per fold)',
                 f"{ta.get('mean_epoch_time', (np.nan,))[0]:.2f}",
                 f"{tns.get('mean_epoch_time', (np.nan,))[0]:.2f}",
                 f"{100*ta.get('mean_refined_frac', (np.nan,))[0]:.0f}%", '100%',
                 f"{ta.get('train_time', (np.nan,))[0]:.0f}"])
S5.append(P('Table 8 reports the computational profile of adaptive structure learning (all timings on CPU). The complete AHGSL model trains in minutes on every benchmark despite jointly optimizing the topology, and the overhead of graph adaptation is controlled by SER: whereas the unrestricted variant recommits the entire candidate edge space at every epoch, the full model touches only the selected fraction of edges per epoch on average, and this fraction decays over training as the structure stabilizes (Figure 3b). The adaptive components therefore add only a modest cost over the static heterogeneous baseline while providing the accuracy improvements of Table 4 and the analysis capabilities of Sections 5.8–5.9.'))
S5.append(CAP('Table 8. Computational profile (mean per-epoch wall-clock time in seconds; structural updates as mean fraction of candidate edges committed per epoch)'))
S5.append(TBL(['Dataset', 'Epoch time AHGSL (s)', 'Epoch time w/o SER (s)',
               'Committed edges AHGSL', 'Committed edges w/o SER',
               'Total time (s)'], eff_rows))

S6 = []
S6.append(P('This paper introduced AHGSL, an adaptive heterogeneous graph structure learning framework for joint gene selection and multi-label classification of high-dimensional gene-expression data. Unlike conventional graph-based pipelines that fix the topology before training, AHGSL treats the heterogeneous graph connecting genes, samples, and labels as a learnable object: the Adaptive Edge Evolution mechanism re-estimates relation-specific edge weights from the evolving node representations, Selective Edge Refinement restricts structural updates to high-impact and uncertain relations, and a gated gene-selection component is optimized jointly with the multi-label objective and a graph-stability constraint.'))
S6.append(P('Comprehensive experiments on real public benchmarks — five ready-made multi-label yeast gene-expression benchmarks (four FunCat datasets and the canonical Mulan yeast-14 dataset) and the TCGA Pan-Cancer RNA-seq cohort (801 samples × 20,531 genes) — support the design. Adaptive topology learning matched or exceeded an otherwise identical static heterogeneous model on micro-AUC on all five yeast benchmarks '
            + f'(mean gains of {100*f1_gain_static:.1f} micro-F1 and {100*ap_gain_static:.1f} AP points, and the best overall results on the canonical yeast-14 benchmark), '
            'with ablations attributing the largest accuracy contribution to AEE, while SER provided an order-of-magnitude reduction in structural updates at no accuracy cost. The analysis of the learned structure showed the intended behavior: refinement activity decays as edge scores polarize, weak relations are pruned, new candidate relations are activated, and the final topology consists of strongly weighted informative relations rather than a uniformly rewired graph. On TCGA, the framework scaled to the full expression space, attained near-ceiling accuracy comparable to strong baselines '
            + f'({ms(tcga_acc, 4)} accuracy, {ms(tcga_auc, 4)} macro-AUC), '
            + f'and its jointly learned gate produced compact biomarker panels without any separate selection stage: 50 gate-selected genes support {gate50:.1f}% cross-validated accuracy, and 100 genes come within half a point of the full feature space, while classical univariate filters remain strong and somewhat more fold-stable on this highly separable benchmark.'))
S6.append(P('Several directions remain open. First, the framework should be evaluated on human cohorts with validated sample-level multi-label structure — for example the multiCRIS colorectal-cancer formulation, in which individual TCGA-COAD/READ tumors legitimately carry several transcriptional subtype labels simultaneously — which constitutes the most direct clinical instantiation of the gene–sample–label design; the present pipeline applies to such data without modification. Second, prior biological knowledge (protein–protein interaction and pathway networks) can be injected into the initial topology and candidate pool, allowing AEE to correct rather than replace curated structure. Third, scaling the candidate-edge space with locality-sensitive or learned sparsification would extend the framework to single-cell resolution, and integrating identifier-level gene annotations would enable direct biological interpretation of the selected panels. Finally, extending the stability analysis with formal guarantees on the convergence of the alternating structure/parameter optimization is an interesting theoretical direction.'))

REFS_DATA = []
REFS_DATA.append(H('References for Section 5 (Experimental Study)', sz=22))
for t in [
    '[1] Vens C., Struyf J., Schietgat L., Džeroski S., Blockeel H. (2008). Decision trees for hierarchical multi-label classification. Machine Learning, 73(2), 185–214. (Yeast FunCat benchmark collection: cellcycle, church, derisi, eisen; datasets distributed with the Clus-HMC software.)',
    '[2] Weinstein J.N., Collisson E.A., Mills G.B., et al.; The Cancer Genome Atlas Research Network. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. Nature Genetics, 45(10), 1113–1120.',
    '[3] Fiorini S. (2016). Gene expression cancer RNA-Seq data set (TCGA Pan-Cancer, 801 samples × 20,531 genes). UCI Machine Learning Repository.',
    '[4] Kipf T.N., Welling M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.',
    '[5] Wang X., Ji H., Shi C., et al. (2019). Heterogeneous graph attention network. Proceedings of the WWW Conference, 2022–2032.',
    '[6] Spellman P.T., Sherlock G., Zhang M.Q., et al. (1998). Comprehensive identification of cell cycle-regulated genes of the yeast Saccharomyces cerevisiae by microarray hybridization. Molecular Biology of the Cell, 9(12), 3273–3297.',
    '[7] DeRisi J.L., Iyer V.R., Brown P.O. (1997). Exploring the metabolic and genetic control of gene expression on a genomic scale. Science, 278(5338), 680–686.',
    '[8] Eisen M.B., Spellman P.T., Brown P.O., Botstein D. (1998). Cluster analysis and display of genome-wide expression patterns. PNAS, 95(25), 14863–14868.',
    '[9] Elisseeff A., Weston J. (2002). A kernel method for multi-labelled classification. Advances in Neural Information Processing Systems (NIPS), 14, 681–687. (Canonical multi-label yeast dataset; distributed with the Mulan library.)',
]:
    REFS_DATA.append(P(t, align='left', space_after=100))

NEW_ABSTRACT = ('Comprehensive experiments are conducted on real, publicly available benchmarks: five ready-made multi-label yeast gene-expression benchmarks (the four FunCat datasets — cellcycle, church, derisi, eisen — and the canonical 14-label Mulan yeast dataset) and the TCGA Pan-Cancer RNA-seq cohort (801 tumor samples × 20,531 genes, five cancer types). On the multi-label benchmarks, AHGSL is the strongest graph-based method throughout and matches or exceeds an otherwise identical static heterogeneous model on micro-AUC across all five datasets '
                + f'(mean gains of {100*f1_gain_static:.1f} micro-F1 and {100*ap_gain_static:.1f} average-precision points), '
                'attaining the best results of all compared methods on the canonical yeast-14 benchmark; ablations attribute the largest accuracy contribution to adaptive edge evolution. On TCGA, the framework scales to the full expression space, attains '
                + f'{ms(tcga_acc, 4)} cross-validated accuracy, '
                + f'and its jointly learned gene gate yields compact biomarker panels as a by-product of training: 50 selected genes support {gate50:.1f}% accuracy and 100 genes (under 0.5% of the transcriptome) come within half a point of full-feature-space performance. '
                'Analysis of the learned topology confirms the intended behavior: structural refinement activity decays as edge scores polarize, weak relations are pruned and informative ones strengthened, and the resulting graphs remain stable. The proposed adaptive graph learning paradigm thus provides a flexible and scalable solution for high-dimensional biomedical data analysis and a practical route toward graph-based precision medicine applications.')


# ================================================================= ASSEMBLY
def main():
    import subprocess
    orig = os.path.join(BASE, 'manuscript', 'original.docx')
    if os.path.exists(UNP):
        shutil.rmtree(UNP)
    subprocess.run(['unzip', '-q', orig, '-d', UNP], check=True)
    subprocess.run(['find', UNP, '-type', 'l', '-delete'])
    subprocess.run(['python3', '/root/.claude/skills/synced/ce4ea7db-c650-46da-8076-929abd88c180_ad270d5d-57b1-43d4-a25c-8ad6bc7aa50c/docx/scripts/merge_runs.py', UNP], check=True)
    for srcp, name in _fig_files:
        shutil.copy(srcp, os.path.join(UNP, 'word', 'media', name))
    doc_p = os.path.join(UNP, 'word', 'document.xml')
    tree = etree.parse(doc_p)
    body = tree.getroot().find('w:body', NS)
    ch = list(body)

    def ptext(p):
        return ''.join(t.text or '' for t in p.findall('.//w:t', NS))

    # --- crop the burned-in "Fig. X." caption from the architecture figure ---
    media_dir = os.path.join(UNP, 'word', 'media')
    # find rel target of the image in element 130
    blips = ch[130].findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
    if blips:
        rid = blips[0].get(f'{{{R}}}embed')
        rels_p = os.path.join(UNP, 'word', '_rels', 'document.xml.rels')
        rtree = etree.parse(rels_p)
        for rel in rtree.getroot():
            if rel.get('Id') == rid:
                tgt = os.path.join(UNP, 'word', rel.get('Target'))
                img = Image.open(tgt)
                w_px, h_px = img.size
                img.crop((0, 0, w_px, int(h_px * 0.945))).save(tgt)
                print('cropped architecture figure caption:', rel.get('Target'))

    # --- fix "Fig. X" -> "Fig. 2" in paragraph 129 ---
    for t in ch[129].findall('.//w:t', NS):
        if t.text and 'Fig. X' in t.text:
            t.text = t.text.replace('Fig. X', 'Fig. 2')

    # --- edit abstract paragraph [8]: replace text with completed version ---
    p8 = ch[8]
    ts = p8.findall('.//w:t', NS)
    ts[0].text = NEW_ABSTRACT
    for t in ts[1:]:
        t.text = ''

    # --- build new body order ---
    keep = []
    keep += ch[0:23]           # title..preliminaries intro
    keep += ch[40:69]          # 2.1-2.4 content ([37..39] empties dropped)
    keep += ch[77:123]         # 2.5 + related work
    keep += ch[123:131]        # S4 part A incl architecture image
    # caption for Figure 2 (new)
    fig2cap = X(CAP('Figure 2. Overall architecture of the proposed AHGSL framework.'))
    keep.append(fig2cap)
    keep += ch[214:320]        # 4.1-4.6 equations
    new_s5 = [X(s) for s in S5]
    keep += new_s5
    # conclusion heading exists at [326]
    keep.append(ch[326])
    keep += [X(s) for s in S6]
    keep += ch[350:431]        # references blocks (main + Sections 2-3)
    keep += [X(s) for s in REFS_DATA]
    keep.append(ch[-1])        # sectPr

    for el in list(body):
        body.remove(el)
    for el in keep:
        body.append(el)

    convert_equations(tree)

    # --- update relationships for new images (incl. equation images) ---
    rels_p = os.path.join(UNP, 'word', '_rels', 'document.xml.rels')
    rtree = etree.parse(rels_p)
    rroot = rtree.getroot()
    for rid, tgt in _new_rels:
        e = etree.SubElement(rroot, '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
        e.set('Id', rid)
        e.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
        e.set('Target', tgt)
    rtree.write(rels_p, xml_declaration=True, encoding='UTF-8', standalone=True)

    tree.write(doc_p, xml_declaration=True, encoding='UTF-8', standalone=True)
    print('document assembled:', len(keep), 'body elements;', len(_new_rels), 'new images')


# ===================== OMML equation conversion (Section 4) =====================
CANONICAL_EQS = [
    ('G^{(0)}=(V,E', r'G^{(0)}=\left(V,\;E^{(0)},\;X,\;\Phi,\;\Psi\right)'),
    ('X_G\\in', r'X_G\in\mathbb{R}^{N_G\times d_G}'),
    ('X_S\\in', r'X_S\in\mathbb{R}^{N_S\times d_S}'),
    ('X_L\\in', r'X_L\in\mathbb{R}^{L\times d_L}'),
    ('E^{(0)}=E_{GG}', r'E^{(0)}=E_{GG}\cup E_{SS}\cup E_{LL}\cup E_{SG}\cup E_{SL}'),
    ('a_{ij}^{(0)}=s', r'a_{ij}^{(0)}=s_{\psi}\left(v_i,v_j\right)'),
    ('Concat', r'\tilde{a}_{ij}^{(t+1)}=\sigma\left(w_{\psi}^{\top}\,\mathrm{Concat}\left(h_i^{(t)},\,h_j^{(t)},\,a_{ij}^{(t)}\right)+b_{\psi}\right)'),
    ('0\\leq', r'0\leq\tilde{a}_{ij}^{(t+1)}\leq 1'),
    ('E^{(t+1)}', r'E^{(t+1)}=\left\{\,e_{ij}\in E_c \; : \; \tilde{a}_{ij}^{(t+1)}\geq\eta\,\right\}'),
    ('\\quad\\text{for }', r'a_{ij}^{(t+1)}=\tilde{a}_{ij}^{(t+1)}\qquad\text{for } e_{ij}\in E^{(t+1)}'),
    ('\\Delta a_{ij}', r'\Delta a_{ij}^{(t)}=\left|\,a_{ij}^{(t+1)}-a_{ij}^{(t)}\,\right|'),
    ('U_{ij}^{(t)}', r'U_{ij}^{(t)}=-\,a_{ij}^{(t)}\log a_{ij}^{(t)}-\left(1-a_{ij}^{(t)}\right)\log\left(1-a_{ij}^{(t)}\right)'),
    ('\\mathcal{S}^{(t)}', r'\mathcal{S}^{(t)}=\left\{\,e_{ij}\in E_c \;\mid\; \Delta a_{ij}^{(t)}>\tau \;\vee\; U_{ij}^{(t)}>\theta\,\right\}'),
    ('r_g=', r'r_g=\left\lVert h_g\right\rVert_1'),
    ('TopK', r'\mathcal{G}_K=\mathrm{TopK}\left(\left\{\,r_g : g\in V_G\,\right\},\;K\right)'),
    ('\\hat{y}_{ij}', r'\hat{y}_{ij}=\sigma\left(f_{\theta}\left(h_i^{S},\,h_j^{L},\,\mathcal{G}_K\right)\right)'),
    ('\\mathcal{L}_{cls}', r'\mathcal{L}_{cls}=-\frac{1}{NL}\sum_{i=1}^{N}\sum_{j=1}^{L}\left[\,y_{ij}\log\hat{y}_{ij}+\left(1-y_{ij}\right)\log\left(1-\hat{y}_{ij}\right)\right]'),
    ('\\mathcal{L}_{sel}', r'\mathcal{L}_{sel}=\lambda_g\sum_{g\in V_G}\left\lVert h_g\right\rVert_1'),
    ('\\mathcal{L}_{stab}', r'\mathcal{L}_{stab}=\lambda_s\sum_{e_{ij}\in E_c}\left(a_{ij}^{(t+1)}-a_{ij}^{(t)}\right)^{2}'),
    ('\\mathcal{L}_{total}', r'\mathcal{L}_{total}=\mathcal{L}_{cls}+\mathcal{L}_{sel}+\mathcal{L}_{stab}'),
    ('\\mathcal{L}_{total}', r'\mathcal{L}_{total}=\mathcal{L}_{cls}+\lambda_g\,\mathcal{L}_{sel}+\lambda_s\,\mathcal{L}_{stab}'),
    ('\\rightarrow', r'G^{(0)}\;\rightarrow\;H^{(t)}\;\rightarrow\;\mathcal{G}_K^{(t)}\;\rightarrow\;\mathrm{AEE}\;\rightarrow\;\mathrm{SER}\;\rightarrow\;\mathcal{L}_{total}\;\rightarrow\;G^{(t+1)}'),
]

INLINE_MAP = [
    ('\\operatorname{Concat}(\\cdot)', 'Concat(·)'),
    ('s_{\\psi}(\\cdot)', 's_ψ(·)'),
    ('\\sigma(\\cdot)', 'σ(·)'),
    ('\\psi\\in\\Psi', 'ψ ∈ Ψ'),
    ('\\mathcal{S}^{(t)}', '𝒮^(t)'),
    ('\\mathcal{G}K^*', '𝒢_K*'),
    ('\\mathcal{G}_K', '𝒢_K'),
    ('\\hat{Y}', 'Ŷ'),
    ('w_{\\psi}', 'w_ψ'),
    ('b_{\\psi}', 'b_ψ'),
    ('f_{\\theta}', 'f_θ'),
    ('\\lambda_g', 'λ_g'),
    ('\\lambda_s', 'λ_s'),
    ('\\eta', 'η'),
    ('\\tau', 'τ'),
    ('\\theta', 'θ'),
    ('\\Phi', 'Φ'),
    ('\\Psi', 'Ψ'),
    ('\\psi', 'ψ'),
    ('(G^)', '(G*)'),
    ('^{(t+1)}', '^(t+1)'),
    ('^{(t)}', '^(t)'),
    ('^{(0)}', '^(0)'),
    ('_{ij}', '_ij'),
    ('_{GG}', '_GG'),
    ('_{SS}', '_SS'),
    ('_{LL}', '_LL'),
    ('_{SG}', '_SG'),
    ('_{SL}', '_SL'),
    ('^{S}', '^S'),
    ('^{L}', '^L'),
]


def _ptext(p):
    return ''.join(t.text or '' for t in p.findall('.//w:t', NS))


def _pandoc_omml(latex_list):
    """Convert display LaTeX equations to OMML oMathPara elements via pandoc."""
    import subprocess, zipfile, tempfile
    tex = '\n\n'.join('$$%s$$' % e for e in latex_list)
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, 'eqs.tex')
        dp = os.path.join(td, 'eqs.docx')
        open(tp, 'w').write(tex)
        subprocess.run(['pandoc', '-f', 'latex', tp, '-o', dp], check=True)
        with zipfile.ZipFile(dp) as z:
            doc = z.read('word/document.xml')
    root = etree.fromstring(doc)
    M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
    # pandoc emits some OMML property children out of schema order; fix them
    ORDER = {
        'rPr': ['lit', 'nor', 'scr', 'sty', 'brk', 'aln'],
        'dPr': ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr'],
        'naryPr': ['chr', 'limLoc', 'grow', 'subHide', 'supHide', 'ctrlPr'],
        'oMathParaPr': ['jc'],
    }
    for tag, order in ORDER.items():
        for pr in root.iter('{%s}%s' % (M, tag)):
            kids = list(pr)
            def key(e):
                ln = e.tag.split('}')[1]
                return order.index(ln) if ln in order else len(order)
            for e in sorted(kids, key=key):
                pr.append(e)  # re-append moves into sorted order
    # schema: rPr allows (nor) OR (scr, sty), not both
    for pr in root.iter('{%s}rPr' % M):
        names = {e.tag.split('}')[1] for e in pr}
        if 'nor' in names:
            for e in list(pr):
                if e.tag.split('}')[1] in ('scr', 'sty'):
                    pr.remove(e)
    maths = root.findall('.//{%s}oMathPara' % M)
    assert len(maths) == len(latex_list), f"pandoc produced {len(maths)} equations for {len(latex_list)}"
    return maths


def convert_equations(tree):
    body = tree.getroot().find('w:body', NS)
    ch = list(body)
    i41 = i5 = None
    for i, el in enumerate(ch):
        if el.tag.endswith('}p'):
            t = _ptext(el).strip()
            if t.startswith('4.1 Heterogeneous') and i41 is None:
                i41 = i
            if t == '5. Experimental Results' and i41 is not None:
                i5 = i
                break
    assert i41 and i5, 'section 4 window not found'
    # scan display-equation blocks
    blocks = []  # list of (list_of_elements, joined_text)
    i = i41
    while i < i5:
        el = ch[i]
        if el.tag.endswith('}p'):
            t = _ptext(el)
            ts = t.strip()
            if ts.startswith('[') and ('\\' in ts or '^{' in ts or '_{' in ts):
                els, txt = [el], t
                j = i
                while not txt.rstrip().endswith(']') and j + 1 < i5:
                    j += 1
                    els.append(ch[j])
                    txt += ' ' + _ptext(ch[j])
                blocks.append((els, txt.strip()))
                i = j
        i += 1
    # match each block to a canonical equation, in order
    used = [False] * len(CANONICAL_EQS)
    matched = []
    for els, txt in blocks:
        hit = None
        for k, (anchor, latex) in enumerate(CANONICAL_EQS):
            if used[k]:
                continue
            if anchor in txt:
                hit = k
                break
        assert hit is not None, f'no canonical match for block: {txt[:90]!r}'
        used[hit] = True
        matched.append((els, CANONICAL_EQS[hit][1]))
    assert len(matched) == len(CANONICAL_EQS), \
        f'matched {len(matched)} of {len(CANONICAL_EQS)} equations'
    eq_mode = os.environ.get('EQ_MODE', 'omml')
    W_ = f'{{{W}}}'
    if eq_mode == 'omml':
        maths = _pandoc_omml([lx for _, lx in matched])
        for (els, _), om in zip(matched, maths):
            p = etree.SubElement(els[0].getparent(), W_ + 'p')
            pPr = etree.SubElement(p, W_ + 'pPr')
            sp = etree.SubElement(pPr, W_ + 'spacing')
            sp.set(W_ + 'before', '60'); sp.set(W_ + 'after', '120')
            jc = etree.SubElement(pPr, W_ + 'jc'); jc.set(W_ + 'val', 'center')
            p.append(om)
            els[0].addprevious(p)
            for e in els:
                e.getparent().remove(e)
    else:  # image mode (used for the faithful PDF preview)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['mathtext.fontset'] = 'stix'
        eq_dir = os.path.join(BASE, 'figures', 'eq')
        os.makedirs(eq_dir, exist_ok=True)
        for n, (els, latex) in enumerate(matched):
            mt = (latex.replace('\\lVert', '\\Vert').replace('\\rVert', '\\Vert')
                       .replace('\\text{for }', '\\mathrm{for}\\;\\,')
                       .replace('\\text{', '\\mathrm{').replace('\\mid', '|'))
            png = os.path.join(eq_dir, f'eq_{n:02d}.png')
            fig = plt.figure(figsize=(9, 1.4))
            fig.text(0.5, 0.5, f'${mt}$', ha='center', va='center', fontsize=13)
            fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.06)
            plt.close(fig)
            img = Image.open(png)
            w_in = min(img.size[0] / 300.0, 5.8)
            h_in = w_in * img.size[1] / img.size[0]
            name = f'eq_{n:02d}.png'
            shutil.copy(png, os.path.join(UNP, 'word', 'media', name))
            rid = f'rId{_next_rid[0]}'; _next_rid[0] += 1
            _new_rels.append((rid, f'media/{name}'))
            did = _next_img[0]; _next_img[0] += 1
            p = X(_drawing_p(name, int(w_in * EMU_IN), int(h_in * EMU_IN), rid, did,
                             space_after=120))
            els[0].addprevious(p)
            for e in els:
                e.getparent().remove(e)
    # inline cleanup in the (shifted) window
    body_ch = list(body)
    in_win = False
    for el in body_ch:
        if el.tag.endswith('}p'):
            t = _ptext(el).strip()
            if t.startswith('4.1 Heterogeneous'):
                in_win = True
            if t == '5. Experimental Results':
                in_win = False
            if in_win:
                for tnode in el.findall('.//w:t', NS):
                    if tnode.text:
                        s = tnode.text
                        for a, b in INLINE_MAP:
                            s = s.replace(a, b)
                        tnode.text = s
    print(f'converted {len(matched)} display equations to OMML')


if __name__ == '__main__':
    main()
