"""
export_individual_figures.py
Tạo các figure riêng lẻ từ kết quả OOD analysis đã có.
Dùng cùng kiến trúc WideResNet với run_ood_analysis.py.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# ── Config ──────────────────────────────────────────────────────────────────
BASE    = '/mnt/d/TrustWorthy AI'
CKPT    = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA    = os.path.join(BASE, 'data')
OUT_DIR = os.path.join(BASE, 'results', 'individual')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_TEMP  = 1.0
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

# ── WideResNet (SAME as run_ood_analysis.py) ─────────────────────────────────
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
            BasicBlock(in_ch if i == 0 else out_ch,
                       out_ch, stride if i == 0 else 1, drop)
            for i in range(n)
        ])
    def forward(self, x): return self.net(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, k=10, num_classes=10, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
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

def denormalize(tensor):
    """Denormalize CIFAR-10 image tensor for display."""
    mean = torch.tensor(CIFAR_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR_STD).view(3,1,1)
    img  = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1,2,0).clamp(0,1).numpy()

# ── Load model ───────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
model = WideResNet().to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
model.load_state_dict(state, strict=False)
model.eval()
print("Checkpoint loaded")

# ── Datasets ─────────────────────────────────────────────────────────────────
norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
tf   = transforms.Compose([transforms.ToTensor(), norm])

cifar_test = torchvision.datasets.CIFAR10(DATA, train=False, transform=tf, download=False)
svhn_test  = torchvision.datasets.SVHN(DATA, split='test', transform=tf, download=False)
loader_c = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=2)
loader_s = torch.utils.data.DataLoader(svhn_test,  batch_size=256, shuffle=False, num_workers=2)

# ── Scoring ───────────────────────────────────────────────────────────────────
def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)

def msp_score(logits):
    return -torch.softmax(logits, dim=1).max(dim=1).values

@torch.no_grad()
def compute_scores(loader):
    E_all, M_all, labels_all, logits_all = [], [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        E_all.append(energy_score(logits).cpu())
        M_all.append(msp_score(logits).cpu())
        labels_all.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        logits_all.append(logits.cpu())
    return (torch.cat(E_all).numpy(), torch.cat(M_all).numpy(),
            torch.cat(labels_all).numpy(), torch.cat(logits_all).numpy())

print("Computing scores...")
E_id, M_id, y_id, L_id = compute_scores(loader_c)
E_ood, M_ood, y_ood, L_ood = compute_scores(loader_s)

# Accuracy
preds = L_id.argmax(1)
acc = (preds == y_id).mean() * 100
print(f"CIFAR-10 accuracy: {acc:.2f}%")

# ── Threshold sweep ───────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

labels = np.concatenate([np.zeros(len(E_id)), np.ones(len(E_ood))])   # 0=ID, 1=OOD
E_all_sc = np.concatenate([E_id, E_ood])
M_all_sc = np.concatenate([M_id, M_ood])

thresholds = np.linspace(E_all_sc.min(), E_all_sc.max(), 2000)
fpr_arr, tpr_arr, f1_arr, j_arr = [], [], [], []
for thr in thresholds:
    pred = (E_all_sc > thr).astype(int)
    tp = ((pred==1)&(labels==1)).sum()
    fp = ((pred==1)&(labels==0)).sum()
    fn = ((pred==0)&(labels==1)).sum()
    tn = ((pred==0)&(labels==0)).sum()
    fpr = fp/(fp+tn+1e-8); tpr = tp/(tp+fn+1e-8)
    prec = tp/(tp+fp+1e-8)
    f1  = 2*prec*tpr/(prec+tpr+1e-8)
    fpr_arr.append(fpr); tpr_arr.append(tpr)
    f1_arr.append(f1); j_arr.append(tpr-fpr)

fpr_arr=np.array(fpr_arr); tpr_arr=np.array(tpr_arr)
f1_arr =np.array(f1_arr);  j_arr  =np.array(j_arr)

auroc_E = roc_auc_score(labels, E_all_sc)*100
auroc_M = roc_auc_score(labels, M_all_sc)*100
aupr_E  = average_precision_score(labels, E_all_sc)*100
aupr_M  = average_precision_score(labels, M_all_sc)*100

# FPR95 for energy
idx95 = np.argmin(np.abs(tpr_arr - 0.95))
fpr95_E = fpr_arr[idx95]*100
gamma_fpr95 = thresholds[idx95]

idxF1  = f1_arr.argmax(); gamma_f1 = thresholds[idxF1]
idxJ   = j_arr.argmax();  gamma_j  = thresholds[idxJ]
best_f1 = f1_arr[idxF1]

# MSP thresholds for comparison
M_thresholds = np.linspace(M_all_sc.min(), M_all_sc.max(), 2000)
M_fpr_arr, M_tpr_arr = [], []
for thr in M_thresholds:
    pred = (M_all_sc > thr).astype(int)
    tp = ((pred==1)&(labels==1)).sum(); fp = ((pred==1)&(labels==0)).sum()
    fn = ((pred==0)&(labels==1)).sum(); tn = ((pred==0)&(labels==0)).sum()
    M_fpr_arr.append(fp/(fp+tn+1e-8)); M_tpr_arr.append(tp/(tp+fn+1e-8))
M_fpr_arr=np.array(M_fpr_arr); M_tpr_arr=np.array(M_tpr_arr)
idx95M = np.argmin(np.abs(M_tpr_arr - 0.95))
fpr95_M = M_fpr_arr[idx95M]*100

print("Threshold sweep done")
print(f"Energy: AUROC={auroc_E:.2f}%  FPR95={fpr95_E:.2f}%  BestF1={best_f1:.4f}")
print(f"MSP:    AUROC={auroc_M:.2f}%  FPR95={fpr95_M:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# FIGURE A: Phân phối Energy Score (ID vs OOD)
# ═══════════════════════════════════════════════════════════════════
print("\n[FigA] Energy Score distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-22, 0, 80)
ax.hist(E_id,  bins=bins, density=True, alpha=0.65, color='steelblue',
        label=f'CIFAR-10 (ID)  μ={np.mean(E_id):.2f}, σ={np.std(E_id):.2f}')
ax.hist(E_ood, bins=bins, density=True, alpha=0.65, color='tomato',
        label=f'SVHN (OOD)     μ={np.mean(E_ood):.2f}, σ={np.std(E_ood):.2f}')
ax.axvline(gamma_fpr95, color='green',  ls='--', lw=1.8, label=f'γ_FPR95={gamma_fpr95:.2f}')
ax.axvline(gamma_f1,    color='orange', ls=':',  lw=1.8, label=f'γ_BestF1={gamma_f1:.2f}')
ax.axvline(gamma_j,     color='purple', ls='-.',  lw=1.8, label=f'γ_Youden={gamma_j:.2f}')
ax.set_xlabel('Energy Score E(x)', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title('Phân phối Energy Score: CIFAR-10 (ID) vs SVHN (OOD)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figA_energy_distribution.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE B: Phân phối MSP (ID vs OOD)
# ═══════════════════════════════════════════════════════════════════
print("[FigB] MSP distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
bins_m = np.linspace(-1.0, 0, 80)
ax.hist(M_id,  bins=bins_m, density=True, alpha=0.65, color='steelblue',
        label=f'CIFAR-10 (ID)  μ={np.mean(M_id):.3f}, σ={np.std(M_id):.3f}')
ax.hist(M_ood, bins=bins_m, density=True, alpha=0.65, color='coral',
        label=f'SVHN (OOD)     μ={np.mean(M_ood):.3f}, σ={np.std(M_ood):.3f}')
gamma_msp95 = M_thresholds[idx95M]
ax.axvline(gamma_msp95, color='green', ls='--', lw=1.8,
           label=f'γ_FPR95 (MSP)={gamma_msp95:.3f}')
ax.set_xlabel('MSP Score −max_k p(y=k|x)', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title('Phân phối MSP Score: CIFAR-10 (ID) vs SVHN (OOD)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figB_msp_distribution.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE C: ROC Curve so sánh Energy vs MSP
# ═══════════════════════════════════════════════════════════════════
print("[FigC] ROC Curve...")
# sort for plotting
sortE = np.argsort(fpr_arr); sortM = np.argsort(M_fpr_arr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_arr[sortE], tpr_arr[sortE], color='steelblue', lw=2.2,
        label=f'Energy Score  (AUROC = {auroc_E:.2f}%)')
ax.plot(M_fpr_arr[sortM], M_tpr_arr[sortM], color='tomato', lw=2.2, ls='--',
        label=f'MSP Baseline  (AUROC = {auroc_M:.2f}%)')
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.4, label='Random (50%)')
ax.scatter([fpr95_E/100],[0.95], color='steelblue', s=80, zorder=5,
           label=f'FPR95(Energy)={fpr95_E:.1f}%')
ax.scatter([fpr95_M/100],[0.95], color='tomato',    s=80, marker='D', zorder=5,
           label=f'FPR95(MSP)={fpr95_M:.1f}%')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curve: Energy Score vs MSP', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figC_roc_curve.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE D: FPR & TPR theo ngưỡng γ (Energy)
# ═══════════════════════════════════════════════════════════════════
print("[FigD] FPR & TPR vs threshold...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, fpr_arr, color='tomato',    lw=2, label='FPR (↓ tốt hơn)')
ax.plot(thresholds, tpr_arr, color='steelblue', lw=2, label='TPR (↑ tốt hơn)')
ax.axvline(gamma_fpr95, color='green',  ls='--', lw=1.5, label=f'γ_FPR95={gamma_fpr95:.2f}')
ax.axvline(gamma_f1,    color='orange', ls=':',  lw=1.5, label=f'γ_BestF1={gamma_f1:.2f}')
ax.axvline(gamma_j,     color='purple', ls='-.', lw=1.5, label=f'γ_Youden={gamma_j:.2f}')
ax.axhline(0.95, color='grey', ls=':', lw=1, alpha=0.6)
ax.set_xlabel('Ngưỡng phân loại γ (Energy Score)', fontsize=12)
ax.set_ylabel('Tỷ lệ', fontsize=12)
ax.set_title('FPR và TPR theo ngưỡng γ (Energy Score)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figD_fpr_tpr_vs_threshold.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE E: F1 và Youden's J theo ngưỡng γ
# ═══════════════════════════════════════════════════════════════════
print("[FigE] F1 & Youden's J vs threshold...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_arr, color='steelblue', lw=2,   label="Best F1")
ax.plot(thresholds, j_arr,  color='darkorange', lw=2, ls='--', label="Youden's J = TPR−FPR")
ax.axvline(gamma_f1, color='steelblue', ls=':', lw=1.5,
           label=f'γ_F1={gamma_f1:.2f}  F1={best_f1:.4f}')
ax.axvline(gamma_j,  color='darkorange', ls=':', lw=1.5,
           label=f'γ_J={gamma_j:.2f}  J={j_arr[idxJ]:.4f}')
ax.set_xlabel('Ngưỡng phân loại γ (Energy Score)', fontsize=12)
ax.set_ylabel('Giá trị chỉ số', fontsize=12)
ax.set_title("F1-Score và Youden's J theo ngưỡng γ", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figE_f1_youden_vs_threshold.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE F: Precision-Recall Curve so sánh
# ═══════════════════════════════════════════════════════════════════
print("[FigF] Precision-Recall curve...")
from sklearn.metrics import precision_recall_curve
prec_E, rec_E, _ = precision_recall_curve(labels, E_all_sc)
prec_M, rec_M, _ = precision_recall_curve(labels, M_all_sc)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_E, prec_E, color='steelblue', lw=2.2,
        label=f'Energy Score  (AUPR = {aupr_E:.2f}%)')
ax.plot(rec_M, prec_M, color='tomato',    lw=2.2, ls='--',
        label=f'MSP Baseline  (AUPR = {aupr_M:.2f}%)')
ax.set_xlabel('Recall (TPR)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve: Energy Score vs MSP', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.3, 1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'figF_precision_recall.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE G1-G4: 4 ảnh mẫu CIFAR-10 (ID) riêng lẻ
# ═══════════════════════════════════════════════════════════════════
print("[FigG] Individual ID samples (CIFAR-10)...")
cifar_raw = torchvision.datasets.CIFAR10(DATA, train=False, download=False)

chosen_id = []
used_classes = set()
for idx in range(len(cifar_test)):
    img_t, _ = cifar_test[idx]
    _, label = cifar_raw[idx]
    if label not in used_classes and len(chosen_id) < 4:
        img_n = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_n)
            e_sc = energy_score(logits).item()
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, 1).max().item()
        chosen_id.append({
            'img': denormalize(img_t.unsqueeze(0)),
            'true': CIFAR10_CLASSES[label],
            'pred': CIFAR10_CLASSES[pred],
            'energy': e_sc, 'conf': conf, 'label': label
        })
        used_classes.add(label)
    if len(chosen_id) == 4: break

for i, item in enumerate(chosen_id):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(item['img'], 0, 1))
    ax.axis('off')
    color = 'steelblue'
    title = (f"[ID] CIFAR-10 – True: {item['true']}\n"
             f"Pred: {item['pred']} | Conf: {item['conf']:.3f}\n"
             f"Energy Score: {item['energy']:.3f}")
    ax.set_title(title, fontsize=11, color=color, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'figG{i+1}_id_sample_{item["true"]}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE H1-H4: 4 ảnh mẫu SVHN (OOD) riêng lẻ
# ═══════════════════════════════════════════════════════════════════
print("[FigH] Individual OOD samples (SVHN)...")
tf_svhn_raw = T.Compose([T.ToTensor()])
svhn_raw = torchvision.datasets.SVHN(DATA, split='test', transform=tf_svhn_raw, download=False)

chosen_ood = []
used_digits = set()
for idx in range(len(svhn_raw)):
    img_t, label = svhn_raw[idx]
    digit = int(label)
    if digit not in used_digits and len(chosen_ood) < 4:
        img_norm = norm(img_t).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_norm)
            e_sc = energy_score(logits).item()
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, 1).max().item()
        chosen_ood.append({
            'img': img_t.permute(1,2,0).numpy(),
            'true_digit': digit,
            'pred': CIFAR10_CLASSES[pred],
            'energy': e_sc, 'conf': conf
        })
        used_digits.add(digit)
    if len(chosen_ood) == 4: break

for i, item in enumerate(chosen_ood):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(item['img'], 0, 1))
    ax.axis('off')
    color = 'tomato'
    title = (f"[OOD] SVHN – True digit: {item['true_digit']}\n"
             f"Pred (sai): {item['pred']} | Conf: {item['conf']:.3f}\n"
             f"Energy Score: {item['energy']:.3f}")
    ax.set_title(title, fontsize=11, color=color, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'figH{i+1}_ood_sample_digit{item["true_digit"]}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

print("\n=== ALL FIGURES EXPORTED TO:", OUT_DIR, "===")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)
