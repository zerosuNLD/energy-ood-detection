"""
run_ood_analysis.py
Chạy toàn bộ pipeline Energy-based OOD Detection:
  1. Load/train WideResNet-28-10 trên CIFAR-10
  2. Tính Energy Score và MSP cho CIFAR-10 (ID) + SVHN (OOD)
  3. Threshold sweep, ROC, PR curves
  4. Lưu ảnh kết quả ra thư mục /mnt/d/TrustWorthy AI/results/
  5. In số liệu để copy vào LaTeX
"""

import os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_curve, auc,
                             precision_recall_curve,
                             average_precision_score)

# ── Paths ────────────────────────────────────────────────────────
OUT_DIR   = '/mnt/d/TrustWorthy AI/results'
DATA_DIR  = '/mnt/d/TrustWorthy AI/data'
CKPT      = '/mnt/d/TrustWorthy AI/wrn28_10_cifar10.pth'
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
BATCH  = 128
T      = 1.0

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ================================================================
# Model: WideResNet-28-10
# ================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop=0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
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

# ================================================================
# Data
# ================================================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

print("Downloading datasets...")
cifar_train = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
cifar_test  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
svhn_test   = torchvision.datasets.SVHN(  DATA_DIR, split='test', download=True, transform=test_tf)

train_ldr = DataLoader(cifar_train, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
id_ldr    = DataLoader(cifar_test,  batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
ood_ldr   = DataLoader(svhn_test,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

print(f"CIFAR-10 train={len(cifar_train):,}  test={len(cifar_test):,}")
print(f"SVHN test={len(svhn_test):,}")

# ================================================================
# Train / Load model
# ================================================================
def train(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = nn.CrossEntropyLoss()(out, labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(imgs)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(imgs)
        scheduler.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:2d}/{epochs}: loss={loss_sum/total:.4f}  acc={100*correct/total:.2f}%",
                  flush=True)
    return model

model = WideResNet(depth=28, k=10).to(DEVICE)

if os.path.exists(CKPT):
    print(f"Loading checkpoint: {CKPT}")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
else:
    print(f"Training WideResNet-28-10 for {EPOCHS} epochs...")
    model = train(model, train_ldr, EPOCHS)
    torch.save(model.state_dict(), CKPT)
    print(f"Saved → {CKPT}")

# ── Accuracy ─────────────────────────────────────────────────────
model.eval()
ok = tot = 0
with torch.no_grad():
    for x, y in id_ldr:
        p = model(x.to(DEVICE)).argmax(1)
        ok  += (p == y.to(DEVICE)).sum().item()
        tot += len(y)
cifar_acc = 100 * ok / tot
print(f"CIFAR-10 test accuracy: {cifar_acc:.2f}%")

# ================================================================
# Scoring functions
# ================================================================
def get_scores(model, loader, mode='energy', temperature=1.0):
    model.eval()
    out = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(DEVICE))
            if mode == 'energy':
                s = -temperature * torch.logsumexp(logits / temperature, dim=1)
            else:
                s = -F.softmax(logits, dim=1).max(dim=1).values
            out.append(s.cpu().numpy())
    return np.concatenate(out)

print("Computing scores...")
e_id  = get_scores(model, id_ldr,  mode='energy', temperature=T)
e_ood = get_scores(model, ood_ldr, mode='energy', temperature=T)
m_id  = get_scores(model, id_ldr,  mode='msp')
m_ood = get_scores(model, ood_ldr, mode='msp')

print(f"Energy  ID  mean={e_id.mean():.3f}  std={e_id.std():.3f}")
print(f"Energy  OOD mean={e_ood.mean():.3f}  std={e_ood.std():.3f}")
print(f"MSP     ID  mean={m_id.mean():.3f}  std={m_id.std():.3f}")
print(f"MSP     OOD mean={m_ood.mean():.3f}  std={m_ood.std():.3f}")

# ================================================================
# Threshold sweep & metrics
# ================================================================
def sweep(scores_id, scores_ood, n=2000):
    all_scores = np.concatenate([scores_id, scores_ood])
    all_labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n)
    fprs = np.empty(n); tprs = np.empty(n)
    precs = np.empty(n); f1s = np.empty(n)
    for i, g in enumerate(thresholds):
        pred = (all_scores >= g).astype(int)
        TP = ((pred == 1) & (all_labels == 1)).sum()
        FP = ((pred == 1) & (all_labels == 0)).sum()
        TN = ((pred == 0) & (all_labels == 0)).sum()
        FN = ((pred == 0) & (all_labels == 1)).sum()
        tprs[i]  = TP / (TP + FN + 1e-9)
        fprs[i]  = FP / (FP + TN + 1e-9)
        precs[i] = TP / (TP + FP + 1e-9)
        f1s[i]   = 2*precs[i]*tprs[i] / (precs[i] + tprs[i] + 1e-9)
    return thresholds, fprs, tprs, precs, f1s

def fpr_at_tpr(scores_id, scores_ood, tpr_target=0.95):
    labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])
    fpr_v, tpr_v, thr_v = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr_v - tpr_target))
    return thr_v[idx], fpr_v[idx], tpr_v[idx]

print("Sweeping thresholds...")
th_e, fpr_e, tpr_e, prec_e, f1_e = sweep(e_id, e_ood)
th_m, fpr_m, tpr_m, prec_m, f1_m = sweep(m_id, m_ood)

g_fpr95_e, fpr95_e, _ = fpr_at_tpr(e_id, e_ood)
g_fpr95_m, fpr95_m, _ = fpr_at_tpr(m_id, m_ood)

g_f1_e = th_e[np.argmax(f1_e)]
g_f1_m = th_m[np.argmax(f1_m)]

youden_e   = tpr_e - fpr_e
g_youden_e = th_e[np.argmax(youden_e)]

youden_m   = tpr_m - fpr_m
g_youden_m = th_m[np.argmax(youden_m)]

# ROC / PR
lbl_e = np.concatenate([np.zeros(len(e_id)), np.ones(len(e_ood))])
sc_e  = np.concatenate([e_id, e_ood])
lbl_m = np.concatenate([np.zeros(len(m_id)), np.ones(len(m_ood))])
sc_m  = np.concatenate([m_id, m_ood])

fpr_roc_e, tpr_roc_e, _ = roc_curve(lbl_e, sc_e)
fpr_roc_m, tpr_roc_m, _ = roc_curve(lbl_m, sc_m)
auroc_e = auc(fpr_roc_e, tpr_roc_e)
auroc_m = auc(fpr_roc_m, tpr_roc_m)

prec_pr_e, rec_pr_e, _ = precision_recall_curve(lbl_e, sc_e)
prec_pr_m, rec_pr_m, _ = precision_recall_curve(lbl_m, sc_m)
ap_e = average_precision_score(lbl_e, sc_e)
ap_m = average_precision_score(lbl_m, sc_m)

# ================================================================
# Print summary numbers
# ================================================================
sep = '=' * 65
print(f'\n{sep}')
print('   KẾT QUẢ THỰC NGHIỆM — ENERGY OOD DETECTION')
print(f'{sep}')
print(f'  CIFAR-10 test accuracy    : {cifar_acc:.2f}%')
print(f'{"-"*65}')
print(f'{"Metric":<28} {"Energy Score":>16} {"MSP (baseline)":>16}')
print(f'{"-"*65}')
print(f'{"FPR95 (%)":<28} {fpr95_e*100:>16.2f} {fpr95_m*100:>16.2f}')
print(f'{"AUROC (%)":<28} {auroc_e*100:>16.2f} {auroc_m*100:>16.2f}')
print(f'{"Best F1":<28} {f1_e.max():>16.4f} {f1_m.max():>16.4f}')
print(f'{"AUPR (Avg Precision)":<28} {ap_e:>16.4f} {ap_m:>16.4f}')
print(f'{sep}')
print(f'\nEnergy ID  : mean={e_id.mean():.3f}  std={e_id.std():.3f}  '
      f'min={e_id.min():.3f}  max={e_id.max():.3f}')
print(f'Energy OOD : mean={e_ood.mean():.3f}  std={e_ood.std():.3f}  '
      f'min={e_ood.min():.3f}  max={e_ood.max():.3f}')
print(f'\nNgưỡng Energy Score tối ưu:')
print(f'  ① γ_FPR95  = {g_fpr95_e:.4f}  → FPR={fpr95_e*100:.2f}%  TPR=95.00%')
print(f'  ② γ_BestF1 = {g_f1_e:.4f}  → F1={f1_e.max():.4f}  '
      f'FPR={fpr_e[np.argmax(f1_e)]*100:.2f}%  TPR={tpr_e[np.argmax(f1_e)]*100:.2f}%')
print(f'  ③ γ_Youden = {g_youden_e:.4f}  → J={youden_e.max():.4f}  '
      f'FPR={fpr_e[np.argmax(youden_e)]*100:.2f}%  TPR={tpr_e[np.argmax(youden_e)]*100:.2f}%')
print(f'{sep}\n')

# ================================================================
# FIGURE 1: Phân tích ngưỡng tổng hợp (7 biểu đồ)
# ================================================================
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

BLUE   = '#1976D2'; RED    = '#D32F2F'; ORANGE = '#F57C00'
GREEN  = '#388E3C'; PURPLE = '#7B1FA2'; TEAL   = '#00796B'

fig = plt.figure(figsize=(20, 15))
fig.suptitle(
    'Energy-based OOD Detection — Phân Tích Ngưỡng (Threshold Analysis)\n'
    f'CIFAR-10 (In-Distribution)  vs  SVHN (Out-of-Distribution)   '
    f'[WRN-28-10, CIFAR-10 acc={cifar_acc:.1f}%]',
    fontsize=14, fontweight='bold', y=1.005
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── ① Phân phối Energy Score ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(e_id,  bins=120, alpha=0.60, color=BLUE,  density=True,
         label=f'CIFAR-10 – In-Distribution  (mean={e_id.mean():.2f})')
ax1.hist(e_ood, bins=120, alpha=0.60, color=RED,   density=True,
         label=f'SVHN – Out-of-Distribution  (mean={e_ood.mean():.2f})')
ax1.axvline(g_fpr95_e,  color=GREEN,  lw=2.0, ls='--',
            label=f'γ₁ (TPR=95%) = {g_fpr95_e:.2f}')
ax1.axvline(g_f1_e,     color=PURPLE, lw=2.0, ls=':',
            label=f'γ₂ (Best F1) = {g_f1_e:.2f}')
ax1.axvline(g_youden_e, color=TEAL,   lw=2.0, ls='-.',
            label=f'γ₃ (Youden)  = {g_youden_e:.2f}')
ax1.set_xlabel('Energy Score  E(x)', fontsize=11)
ax1.set_ylabel('Mật độ xác suất', fontsize=11)
ax1.set_title('① Phân phối Energy Score — In-Dist vs OOD\n'
              '(Khoảng tách biệt càng lớn → phát hiện OOD càng dễ)',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=8.5)

# ── ② ROC Curve ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(fpr_roc_e, tpr_roc_e, color=BLUE,   lw=2,
         label=f'Energy  AUROC={auroc_e:.4f}')
ax2.plot(fpr_roc_m, tpr_roc_m, color=ORANGE, lw=2, ls='--',
         label=f'MSP     AUROC={auroc_m:.4f}')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.35, label='Random (0.5000)')
ax2.scatter([fpr95_e], [0.95], color=GREEN,  s=90, zorder=6,
            label=f'FPR95(E)={fpr95_e:.4f}')
ax2.scatter([fpr95_m], [0.95], color=ORANGE, s=90, marker='^', zorder=6,
            label=f'FPR95(M)={fpr95_m:.4f}')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title(f'② ROC Curve\n(Energy AUROC={auroc_e:.4f}  vs  MSP={auroc_m:.4f})',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)

# ── ③ FPR & TPR vs Threshold ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(th_e, fpr_e, color=RED,   lw=2.2,
         label='FPR — tỷ lệ ID bị nhầm OOD (↓ muốn thấp)')
ax3.plot(th_e, tpr_e, color=GREEN, lw=2.2,
         label='TPR — tỷ lệ OOD phát hiện đúng (↑ muốn cao)')
ax3.axhline(0.95, color='gray', lw=1, ls=':', alpha=0.6)
ax3.text(th_e.max() * 0.99, 0.96, 'TPR = 95%', ha='right', fontsize=8, color='gray')
ax3.axvline(g_fpr95_e,  color=GREEN,  lw=2, ls='--',
            label=f'① γ={g_fpr95_e:.2f}  (TPR=95%, FPR={fpr95_e:.3f})')
ax3.axvline(g_f1_e,     color=PURPLE, lw=2, ls=':',
            label=f'② γ={g_f1_e:.2f}  (Best F1={f1_e.max():.4f})')
ax3.axvline(g_youden_e, color=TEAL,   lw=2, ls='-.',
            label=f'③ γ={g_youden_e:.2f}  (Youden J={youden_e.max():.4f})')
ax3.fill_between(th_e, 0, 1,
                 where=(th_e >= g_fpr95_e - 0.5) & (th_e <= g_fpr95_e + 0.5),
                 alpha=0.10, color=GREEN, label='Vùng ngưỡng tối ưu (FPR95)')
ax3.set_xlabel('Ngưỡng  γ  (Energy Score)', fontsize=11)
ax3.set_ylabel('Tỷ lệ', fontsize=11)
ax3.set_title('③ FPR & TPR theo Ngưỡng γ\n'
              '(Ngưỡng nhỏ → nhiều False Alarm | Ngưỡng lớn → bỏ sót OOD)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=8.5, loc='center right')
ax3.set_xlim(th_e.min(), th_e.max())
ax3.set_ylim(-0.02, 1.05)

# ── ④ F1 & Youden's J vs Threshold ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(th_e, f1_e,     color=PURPLE, lw=2,
         label=f'F1  (max={f1_e.max():.4f})')
ax4.plot(th_e, youden_e, color=TEAL,   lw=2, ls='--',
         label=f"Youden J (max={youden_e.max():.4f})")
ax4.axvline(g_f1_e,     color=PURPLE, lw=1.5, ls=':',
            label=f'Best F1 → γ={g_f1_e:.2f}')
ax4.axvline(g_youden_e, color=TEAL,   lw=1.5, ls='-.',
            label=f'Best J  → γ={g_youden_e:.2f}')
ax4.set_xlabel('Ngưỡng  γ', fontsize=11)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title(f'④ F1 & Youden\'s J theo Ngưỡng\n'
              f'(max F1={f1_e.max():.4f}, max J={youden_e.max():.4f})',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=8.5)

# ── ⑤ Precision-Recall Curve ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(rec_pr_e, prec_pr_e, color=BLUE,   lw=2,
         label=f'Energy  AUPR={ap_e:.4f}')
ax5.plot(rec_pr_m, prec_pr_m, color=ORANGE, lw=2, ls='--',
         label=f'MSP     AUPR={ap_m:.4f}')
ax5.set_xlabel('Recall', fontsize=11)
ax5.set_ylabel('Precision', fontsize=11)
ax5.set_title(f'⑤ Precision-Recall Curve\n(AUPR: Energy={ap_e:.4f}  MSP={ap_m:.4f})',
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)

# ── ⑥ So sánh tổng hợp (bar chart) ──────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
metric_names = ['FPR95 (%)\n↓ thấp hơn tốt', 'AUROC (%)\n↑ cao hơn tốt',
                'Best F1 (%)\n↑ cao hơn tốt', 'AUPR (%)\n↑ cao hơn tốt']
e_vals = [fpr95_e*100, auroc_e*100, f1_e.max()*100, ap_e*100]
m_vals = [fpr95_m*100, auroc_m*100, f1_m.max()*100, ap_m*100]
x = np.arange(len(metric_names)); w = 0.35
b1 = ax6.bar(x - w/2, e_vals, w, color=BLUE,   alpha=0.85, label='Energy Score')
b2 = ax6.bar(x + w/2, m_vals, w, color=ORANGE, alpha=0.85, label='MSP Baseline')
for b, v in zip(list(b1)+list(b2), e_vals+m_vals):
    ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
             f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax6.set_xticks(x); ax6.set_xticklabels(metric_names, fontsize=8)
ax6.set_ylim(0, 120)
ax6.set_title('⑥ So sánh tổng hợp Energy vs MSP\n(cùng mô hình, không train lại)',
              fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)

# ── ⑦ Bảng tóm tắt ngưỡng ────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
rows = [
    ['Tiêu chí chọn γ', 'γ', 'FPR (%)', 'TPR (%)'],
    ['① TPR=95%\n(chuẩn bài báo)',
     f'{g_fpr95_e:.3f}', f'{fpr95_e*100:.2f}', '95.00'],
    ['② Max F1\n(cân bằng P/R)',
     f'{g_f1_e:.3f}',
     f'{fpr_e[np.argmax(f1_e)]*100:.2f}',
     f'{tpr_e[np.argmax(f1_e)]*100:.2f}'],
    ['③ Youden\'s J\n(max TPR-FPR)',
     f'{g_youden_e:.3f}',
     f'{fpr_e[np.argmax(youden_e)]*100:.2f}',
     f'{tpr_e[np.argmax(youden_e)]*100:.2f}'],
]
cell_colors = [['#E3F2FD']*4, ['#E8F5E9']*4, ['#EDE7F6']*4]
tbl = ax7.table(cellText=rows[1:], colLabels=rows[0],
                cellLoc='center', loc='center',
                cellColours=cell_colors)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 2.0)
ax7.set_title('⑦ Tóm tắt 3 tiêu chí\nchọn ngưỡng tối ưu',
              fontsize=11, fontweight='bold', pad=12)

path1 = f'{OUT_DIR}/fig1_threshold_analysis.png'
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")

# ================================================================
# FIGURE 2: Phân phối score chi tiết (Energy vs MSP so sánh)
# ================================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Phân phối Điểm Số: Energy Score vs MSP\n'
              'CIFAR-10 (ID) vs SVHN (OOD)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(e_id,  bins=120, alpha=0.65, color=BLUE, density=True,
        label=f'CIFAR-10 ID\nmean={e_id.mean():.2f}, std={e_id.std():.2f}')
ax.hist(e_ood, bins=120, alpha=0.65, color=RED,  density=True,
        label=f'SVHN OOD\nmean={e_ood.mean():.2f}, std={e_ood.std():.2f}')
ax.axvline(g_fpr95_e, color=GREEN, lw=2, ls='--',
           label=f'γ_FPR95={g_fpr95_e:.2f}')
ax.set_xlabel('Energy Score E(x)', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title('Energy Score Distribution\n'
             f'ΔMean = {e_ood.mean()-e_id.mean():.3f}  (OOD cao hơn ID)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(m_id,  bins=120, alpha=0.65, color=BLUE,   density=True,
        label=f'CIFAR-10 ID\nmean={m_id.mean():.3f}, std={m_id.std():.3f}')
ax.hist(m_ood, bins=120, alpha=0.65, color=ORANGE, density=True,
        label=f'SVHN OOD\nmean={m_ood.mean():.3f}, std={m_ood.std():.3f}')
ax.axvline(g_fpr95_m, color=GREEN, lw=2, ls='--',
           label=f'γ_FPR95={g_fpr95_m:.3f}')
ax.set_xlabel('MSP Score  −max softmax(f)', fontsize=12)
ax.set_ylabel('Mật độ xác suất', fontsize=12)
ax.set_title('MSP Score Distribution\n'
             f'ΔMean = {m_ood.mean()-m_id.mean():.4f}  (tách biệt kém hơn)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

path2 = f'{OUT_DIR}/fig2_score_distributions.png'
plt.tight_layout()
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path2}")

# ================================================================
# FIGURE 3: Mẫu đơn lẻ ID vs OOD
# ================================================================
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def denormalize(tensor):
    mean = torch.tensor(CIFAR_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR_STD).view(3,1,1)
    img  = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1,2,0).clamp(0,1).numpy()

# Lấy mẫu
torch.manual_seed(0)
id_iter  = iter(id_ldr)
id_imgs_b, id_labels_b = next(id_iter)

ood_iter  = iter(ood_ldr)
ood_imgs_b, _ = next(ood_iter)

# Chọn 4 ảnh ID và 4 ảnh OOD để hiển thị
N = 4
fig3, axes3 = plt.subplots(2, N, figsize=(14, 7))
fig3.suptitle('Minh hoạ: In-Distribution (CIFAR-10) vs Out-of-Distribution (SVHN)\n'
              'với Energy Score — Energy CAO = OOD, Energy THẤP = ID',
              fontsize=13, fontweight='bold')

model.eval()
for col in range(N):
    # ID sample
    img_t  = id_imgs_b[col:col+1].to(DEVICE)
    lbl    = id_labels_b[col].item()
    with torch.no_grad():
        logits = model(img_t)
    e_val  = (-T * torch.logsumexp(logits / T, dim=1)).item()
    prob   = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top1   = CIFAR10_CLASSES[np.argmax(prob)]

    ax = axes3[0, col]
    ax.imshow(denormalize(img_t.cpu()))
    ax.axis('off')
    ax.set_title(f'[ID] {CIFAR10_CLASSES[lbl]}\n'
                 f'Pred: {top1}\nE={e_val:.2f}',
                 fontsize=9, color='#1B5E20', fontweight='bold')
    # Viền xanh = ID
    for spine in ax.spines.values():
        spine.set_edgecolor('#1976D2'); spine.set_linewidth(3)

    # OOD sample
    img_t  = ood_imgs_b[col:col+1].to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
    e_val  = (-T * torch.logsumexp(logits / T, dim=1)).item()
    prob   = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top1   = CIFAR10_CLASSES[np.argmax(prob)]
    conf   = prob.max() * 100

    ax = axes3[1, col]
    ax.imshow(denormalize(img_t.cpu()))
    ax.axis('off')
    ax.set_title(f'[OOD] SVHN digit\n'
                 f'Pred(wrong): {top1} ({conf:.0f}%)\nE={e_val:.2f}',
                 fontsize=9, color='#B71C1C', fontweight='bold')
    # Viền đỏ = OOD
    for spine in ax.spines.values():
        spine.set_edgecolor('#D32F2F'); spine.set_linewidth(3)

axes3[0, 0].set_ylabel('In-Distribution\n(CIFAR-10)', fontsize=11,
                        color='#1976D2', fontweight='bold', rotation=90, labelpad=8)
axes3[1, 0].set_ylabel('Out-of-Distribution\n(SVHN)', fontsize=11,
                        color='#D32F2F', fontweight='bold', rotation=90, labelpad=8)

plt.tight_layout()
path3 = f'{OUT_DIR}/fig3_sample_id_vs_ood.png'
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path3}")

# ================================================================
# In số liệu LaTeX
# ================================================================
print("\n" + "="*65)
print("  SỐ LIỆU CHO LATEX")
print("="*65)
print(f"CIFAR-10 acc       = {cifar_acc:.2f}\\%")
print(f"Energy FPR95       = {fpr95_e*100:.2f}\\%")
print(f"MSP    FPR95       = {fpr95_m*100:.2f}\\%")
print(f"Energy AUROC       = {auroc_e*100:.2f}\\%")
print(f"MSP    AUROC       = {auroc_m*100:.2f}\\%")
print(f"Energy Best F1     = {f1_e.max():.4f}")
print(f"MSP    Best F1     = {f1_m.max():.4f}")
print(f"Energy AUPR        = {ap_e:.4f}")
print(f"MSP    AUPR        = {ap_m:.4f}")
print(f"gamma_FPR95        = {g_fpr95_e:.4f}")
print(f"gamma_BestF1       = {g_f1_e:.4f}  FPR={fpr_e[np.argmax(f1_e)]*100:.2f}%  TPR={tpr_e[np.argmax(f1_e)]*100:.2f}%")
print(f"gamma_Youden       = {g_youden_e:.4f}  J={youden_e.max():.4f}  FPR={fpr_e[np.argmax(youden_e)]*100:.2f}%  TPR={tpr_e[np.argmax(youden_e)]*100:.2f}%")
print(f"Energy ID mean/std = {e_id.mean():.3f} / {e_id.std():.3f}")
print(f"Energy OOD mean/std= {e_ood.mean():.3f} / {e_ood.std():.3f}")
print(f"Delta Energy mean  = {e_ood.mean()-e_id.mean():.3f}")
print("="*65)
print("\nAll figures saved to:", OUT_DIR)
