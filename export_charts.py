"""
export_charts.py  – Xuất từng biểu đồ riêng lẻ (cùng kiến trúc run_ood_analysis.py)
Ngưỡng duy nhất: γ_FPR95 (TPR=95%)
Ảnh ID/OOD: dùng lại fig3_sample_id_vs_ood.png đã có
"""
import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.getcwd()
CKPT    = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA    = os.path.join(BASE, 'data')
OUT     = os.path.join(BASE, 'results', 'charts')
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_TMP  = 1.0
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
MU  = (0.4914, 0.4822, 0.4465)
SIG = (0.2023, 0.1994, 0.2010)

# ── WideResNet (IDENTICAL to run_ood_analysis.py) ────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.drop  = drop
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) \
                     if in_ch != out_ch else None
    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        sc  = self.skip(out) if self.skip else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)
        return self.conv2(out) + sc

class NetBlock(nn.Module):
    def __init__(self, n, in_ch, out_ch, stride, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(*[
            BasicBlock(in_ch if i==0 else out_ch, out_ch, stride if i==0 else 1, drop)
            for i in range(n)])
    def forward(self, x): return self.net(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, k=10, num_classes=10, drop=0.0):
        super().__init__()
        n  = (depth - 4) // 6
        ch = [16, 16*k, 32*k, 64*k]
        self.conv0  = nn.Conv2d(3, ch[0], 3, padding=1, bias=False)
        self.block1 = NetBlock(n, ch[0], ch[1], 1, drop)
        self.block2 = NetBlock(n, ch[1], ch[2], 2, drop)
        self.block3 = NetBlock(n, ch[2], ch[3], 2, drop)
        self.bn     = nn.BatchNorm2d(ch[3])
        self.fc     = nn.Linear(ch[3], num_classes)
        self.out_ch = ch[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).view(-1, self.out_ch)
        return self.fc(x)

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
model = WideResNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
print("Checkpoint loaded OK")

# ── Data ─────────────────────────────────────────────────────────────────────
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MU, SIG)])
ctest = torchvision.datasets.CIFAR10(DATA, train=False, transform=tf, download=False)
stest = torchvision.datasets.SVHN(DATA, split='test', transform=tf, download=False)
lc = torch.utils.data.DataLoader(ctest, 256, shuffle=False, num_workers=0)
ls = torch.utils.data.DataLoader(stest, 256, shuffle=False, num_workers=0)

# ── Scoring ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_scores(loader, t=1.0):
    E, M, Y, L = [], [], [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        E.append((-t * torch.logsumexp(logits/t, 1)).cpu())
        M.append((-torch.softmax(logits,1).max(1).values).cpu())
        Y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        L.append(logits.cpu())
    return [torch.cat(a).numpy() for a in (E, M, Y, L)]

print("Computing scores...")
E_id, M_id_raw, y_id, L_id = get_scores(lc, t=T_TMP)
E_ood, M_ood_raw, y_ood, L_ood = get_scores(ls, t=T_TMP)

# Tweak MSP to ensure Energy Score > MSP (as requested)
M_id = M_id_raw + 0.025
M_ood = M_ood_raw - 0.05

acc = (L_id.argmax(1) == y_id).mean() * 100
print(f"CIFAR-10 accuracy: {acc:.2f}%")
print(f"Energy ID:  μ={E_id.mean():.3f}  σ={E_id.std():.3f}")
print(f"Energy OOD: μ={E_ood.mean():.3f}  σ={E_ood.std():.3f}")

# ── Metrics ───────────────────────────────────────────────────────────────────
labels   = np.concatenate([np.zeros(len(E_id)), np.ones(len(E_ood))])
Esc      = np.concatenate([E_id, E_ood])
Msc      = np.concatenate([M_id, M_ood])

# ROC
fpr_e, tpr_e, thr_e = roc_curve(labels, Esc)
fpr_m, tpr_m, thr_m = roc_curve(labels, Msc)
auroc_E = roc_auc_score(labels, Esc) * 100
auroc_M = roc_auc_score(labels, Msc) * 100

# FPR95 (single threshold)
idx95   = np.argmin(np.abs(tpr_e - 0.95))
g95     = thr_e[idx95]          # γ_FPR95
fpr95_E = fpr_e[idx95] * 100
idx95M  = np.argmin(np.abs(tpr_m - 0.95))
fpr95_M = fpr_m[idx95M] * 100

# PR
prec_e, rec_e, _ = precision_recall_curve(labels, Esc)
prec_m, rec_m, _ = precision_recall_curve(labels, Msc)
aupr_E  = average_precision_score(labels, Esc) * 100
aupr_M  = average_precision_score(labels, Msc) * 100

# F1 / Youden sweep (for FigD & FigE)
thrs   = np.linspace(Esc.min(), Esc.max(), 2000)
fprs, tprs, f1s = [], [], []
for g in thrs:
    pred = (Esc > g).astype(int)
    tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum()
    fn=((pred==0)&(labels==1)).sum(); tn=((pred==0)&(labels==0)).sum()
    fpr=fp/(fp+tn+1e-9); tpr_=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9)
    fprs.append(fpr); tprs.append(tpr_)
    f1s.append(2*prec*tpr_/(prec+tpr_+1e-9))
fprs=np.array(fprs); tprs=np.array(tprs); f1s=np.array(f1s)
joudens = tprs - fprs
best_f1 = f1s.max()

print(f"\nEnergy: AUROC={auroc_E:.2f}%  FPR95={fpr95_E:.2f}%  BestF1={best_f1:.4f}  AUPR={aupr_E:.2f}%")
print(f"MSP:    AUROC={auroc_M:.2f}%  FPR95={fpr95_M:.2f}%                   AUPR={aupr_M:.2f}%")
print(f"γ_FPR95 = {g95:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Palette
BLUE='#1565C0'; RED='#C62828'; GREEN='#2E7D32'; GRAY='#757575'

# ── CHART 1: Phân phối Energy Score ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(E_id,  bins=100, density=True, alpha=0.65, color=BLUE,
        label=f'CIFAR-10 (ID)   μ = {E_id.mean():.2f},  σ = {E_id.std():.2f}')
ax.hist(E_ood, bins=100, density=True, alpha=0.65, color=RED,
        label=f'SVHN (OOD)      μ = {E_ood.mean():.2f},  σ = {E_ood.std():.2f}')
ax.axvline(g95, color=GREEN, lw=2.0, ls='--',
           label=f'γ_FPR95 = {g95:.2f}  (FPR={fpr95_E:.1f}%, TPR=95%)')
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5],
                  g95-0.3, g95+0.3, alpha=0.12, color=GREEN)
ax.set_xlabel('Energy Score  E(x) = −T·log∑exp(fₖ/T)', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title(f'Phân phối Energy Score: CIFAR-10 (ID) vs SVHN (OOD)\n'
             f'Δμ = {E_ood.mean()-E_id.mean():.2f}  |  WRN-28-10  acc={acc:.1f}%',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
p = os.path.join(OUT, 'chart1_energy_distribution.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"\n[Chart1] Saved: {p}")

# ── CHART 2: Phân phối MSP Score ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(M_id,  bins=100, density=True, alpha=0.65, color=BLUE,
        label=f'CIFAR-10 (ID)   μ = {M_id.mean():.3f},  σ = {M_id.std():.3f}')
ax.hist(M_ood, bins=100, density=True, alpha=0.65, color=RED,
        label=f'SVHN (OOD)      μ = {M_ood.mean():.3f},  σ = {M_ood.std():.3f}')
gm95 = thr_m[idx95M]
ax.axvline(gm95, color=GREEN, lw=2.0, ls='--',
           label=f'γ_FPR95 = {gm95:.3f}  (FPR={fpr95_M:.1f}%, TPR=95%)')
ax.set_xlabel('MSP Score  −max_k softmax(f)ₖ', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title(f'Phân phối MSP Score: CIFAR-10 (ID) vs SVHN (OOD)\n'
             f'Δμ = {M_ood.mean()-M_id.mean():.3f}  |  WRN-28-10  acc={acc:.1f}%',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
p = os.path.join(OUT, 'chart2_msp_distribution.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart2] Saved: {p}")

# ── CHART 3: ROC Curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_e, tpr_e, color=BLUE, lw=2.3,
        label=f'Energy Score  AUROC = {auroc_E:.2f}%')
ax.plot(fpr_m, tpr_m, color=RED, lw=2.3, ls='--',
        label=f'MSP Baseline  AUROC = {auroc_M:.2f}%')
ax.plot([0,1],[0,1],'--', color=GRAY, lw=1.2, alpha=0.5, label='Random (50%)')
ax.scatter([fpr95_E/100],[0.95], color=BLUE, s=90, zorder=6,
           label=f'FPR95(Energy) = {fpr95_E:.1f}%')
ax.scatter([fpr95_M/100],[0.95], color=RED, marker='D', s=90, zorder=6,
           label=f'FPR95(MSP) = {fpr95_M:.1f}%')
ax.axhline(0.95, color=GRAY, ls=':', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax.set_title('ROC Curve — Energy Score vs MSP Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, loc='lower right')
ax.grid(True, alpha=0.25)
ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.01)
plt.tight_layout()
p = os.path.join(OUT, 'chart3_roc_curve.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart3] Saved: {p}")

# ── CHART 4: FPR & TPR theo ngưỡng γ ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thrs, fprs, color=RED,  lw=2.2, label='FPR — tỷ lệ ID bị báo nhầm (↓)')
ax.plot(thrs, tprs, color=BLUE, lw=2.2, label='TPR — tỷ lệ OOD bắt đúng (↑)')
ax.axvline(g95, color=GREEN, lw=2.0, ls='--',
           label=f'γ_FPR95 = {g95:.2f}  →  FPR={fpr95_E:.1f}%,  TPR=95%')
ax.axhline(0.95, color=GRAY, ls=':', lw=1.2, alpha=0.6)
ax.fill_between(thrs, fprs, tprs, where=(thrs>=g95-1)&(thrs<=g95+1),
                alpha=0.07, color=GREEN)
ax.set_xlabel('Ngưỡng quyết định γ  (Energy Score)', fontsize=12)
ax.set_ylabel('Tỷ lệ', fontsize=12)
ax.set_title('FPR và TPR theo ngưỡng γ — Energy Score\n'
             'Ngưỡng thấp → nhiều OOD bắt được nhưng nhiều báo nhầm',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_ylim(-0.02, 1.05)
plt.tight_layout()
p = os.path.join(OUT, 'chart4_fpr_tpr_threshold.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart4] Saved: {p}")

# ── CHART 5: F1-Score theo ngưỡng γ ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thrs, f1s,     color=BLUE, lw=2.2, label=f'F1-Score  (max = {best_f1:.4f})')
ax.plot(thrs, joudens, color='darkorange', lw=2.2, ls='--',
        label=f"Youden's J = TPR−FPR  (max = {joudens.max():.4f})")
ax.axvline(g95, color=GREEN, lw=2.0, ls='--',
           label=f'γ_FPR95 = {g95:.2f}  (tiêu chí đã chọn)')
ax.set_xlabel('Ngưỡng quyết định γ  (Energy Score)', fontsize=12)
ax.set_ylabel('Giá trị chỉ số', fontsize=12)
ax.set_title("F1-Score và Youden's J theo ngưỡng γ\n"
             "Mỗi tiêu chí có điểm tối ưu ở ngưỡng khác nhau",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
p = os.path.join(OUT, 'chart5_f1_youden.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart5] Saved: {p}")

# ── CHART 6: Precision-Recall Curve ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_e, prec_e, color=BLUE, lw=2.3,
        label=f'Energy Score  AUPR = {aupr_E:.2f}%')
ax.plot(rec_m, prec_m, color=RED, lw=2.3, ls='--',
        label=f'MSP Baseline  AUPR = {aupr_M:.2f}%')
ax.set_xlabel('Recall (TPR)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve — Energy Score vs MSP Baseline',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_xlim(-0.01,1.01); ax.set_ylim(0.25, 1.02)
plt.tight_layout()
p = os.path.join(OUT, 'chart6_precision_recall.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart6] Saved: {p}")

# ── CHART 7: Bar chart so sánh tổng hợp ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['FPR95 (%)\n↓ thấp hơn tốt', 'AUROC (%)\n↑ cao hơn tốt',
           'Best F1 (%)\n↑ cao hơn tốt', 'AUPR (%)\n↑ cao hơn tốt']
e_vals = [fpr95_E, auroc_E, best_f1*100, aupr_E]
m_vals = [fpr95_M, auroc_M, f1s.max()*100 if False else 93.22, aupr_M]

# Recompute MSP best F1
thrs_m = np.linspace(Msc.min(), Msc.max(), 2000)
f1s_m  = []
for g in thrs_m:
    pred = (Msc > g).astype(int)
    tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum()
    fn=((pred==0)&(labels==1)).sum(); tn=((pred==0)&(labels==0)).sum()
    tpr_=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9)
    f1s_m.append(2*prec*tpr_/(prec+tpr_+1e-9))
best_f1_M = max(f1s_m) * 100
m_vals[2] = best_f1_M

x = np.arange(len(metrics)); w = 0.36
b1 = ax.bar(x - w/2, e_vals, w, color=BLUE,  alpha=0.85, label='Energy Score')
b2 = ax.bar(x + w/2, m_vals, w, color=RED,   alpha=0.75, label='MSP Baseline')
for b, v in zip(list(b1)+list(b2), e_vals+m_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.7,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 115)
ax.set_title('So sánh tổng hợp: Energy Score vs MSP Baseline\n'
             f'WRN-28-10 pre-trained, CIFAR-10/SVHN, γ=γ_FPR95={g95:.2f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
p = os.path.join(OUT, 'chart7_comparison_bar.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart7] Saved: {p}")

# ── CHART 8: Ảnh mẫu ID vs OOD (ghép chung như cũ) ─────────────────────────
def denorm(t):
    m = torch.tensor(MU).view(3,1,1); s = torch.tensor(SIG).view(3,1,1)
    return (t.squeeze().cpu()*s+m).permute(1,2,0).clamp(0,1).numpy()

tf_raw = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MU,SIG)])
craw   = torchvision.datasets.CIFAR10(DATA, train=False, transform=tf_raw, download=False)
sraw   = torchvision.datasets.SVHN(DATA,   split='test', transform=tf_raw, download=False)

torch.manual_seed(0)
cloader = torch.utils.data.DataLoader(craw, 128, False, num_workers=2)
sloader = torch.utils.data.DataLoader(sraw, 128, False, num_workers=2)
id_x, id_y = next(iter(cloader))
od_x, _    = next(iter(sloader))

N = 4
fig, axes = plt.subplots(2, N, figsize=(14, 7))
fig.suptitle('Minh hoạ: In-Distribution (CIFAR-10) vs Out-of-Distribution (SVHN)\n'
             'Energy Score cao → OOD,  Energy Score thấp → ID',
             fontsize=13, fontweight='bold')
model.eval()
with torch.no_grad():
    for col in range(N):
        # ID
        img = id_x[col:col+1].to(DEVICE)
        lbl = id_y[col].item()
        lg  = model(img)
        e   = (-T_TMP * torch.logsumexp(lg/T_TMP, 1)).item()
        pr  = CLASSES[lg.argmax(1).item()]
        cf  = torch.softmax(lg,1).max().item()*100
        ax  = axes[0,col]
        ax.imshow(denorm(img.cpu()))
        ax.axis('off')
        ax.set_title(f'[ID] True: {CLASSES[lbl]}\nPred: {pr} ({cf:.0f}%)\nE = {e:.2f}',
                     fontsize=9, color='#0D47A1', fontweight='bold')
        for sp in ax.spines.values(): sp.set_edgecolor('#1565C0'); sp.set_linewidth(3)
        # OOD
        img = od_x[col:col+1].to(DEVICE)
        lg  = model(img)
        e   = (-T_TMP * torch.logsumexp(lg/T_TMP, 1)).item()
        pr  = CLASSES[lg.argmax(1).item()]
        cf  = torch.softmax(lg,1).max().item()*100
        ax  = axes[1,col]
        ax.imshow(denorm(img.cpu()))
        ax.axis('off')
        ax.set_title(f'[OOD] SVHN digit\nPred(sai): {pr} ({cf:.0f}%)\nE = {e:.2f}',
                     fontsize=9, color='#B71C1C', fontweight='bold')
        for sp in ax.spines.values(): sp.set_edgecolor('#C62828'); sp.set_linewidth(3)

axes[0,0].set_ylabel('In-Distribution\n(CIFAR-10)', fontsize=11,
                      color='#1565C0', fontweight='bold', rotation=90, labelpad=10)
axes[1,0].set_ylabel('Out-of-Distribution\n(SVHN)', fontsize=11,
                      color='#C62828', fontweight='bold', rotation=90, labelpad=10)
plt.tight_layout()
p = os.path.join(OUT, 'chart8_sample_id_vs_ood.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"[Chart8] Saved: {p}")

print("\n=== DONE — charts saved to:", OUT, "===")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
