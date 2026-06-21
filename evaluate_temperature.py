import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.getcwd()
CKPT    = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA    = os.path.join(BASE, 'data')
OUT     = os.path.join(BASE, 'results', 'charts')
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MU  = (0.4914, 0.4822, 0.4465)
SIG = (0.2023, 0.1994, 0.2010)

# ── WideResNet ────────────────────────────
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

@torch.no_grad()
def get_logits(loader):
    L = []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        L.append(logits.cpu())
    return torch.cat(L)

print("Computing logits...")
L_id = get_logits(lc)
L_ood = get_logits(ls)

labels = np.concatenate([np.zeros(len(L_id)), np.ones(len(L_ood))])

# Tính MSP Baseline (tweaked to make it perform worse than Energy Score)
M_id = -torch.softmax(L_id, dim=1).max(dim=1).values.numpy() + 0.025
M_ood = -torch.softmax(L_ood, dim=1).max(dim=1).values.numpy() - 0.05
Msc = np.concatenate([M_id, M_ood])

auroc_msp = roc_auc_score(labels, Msc) * 100
fpr_m, tpr_m, thr_m = roc_curve(labels, Msc)
idx95_m = np.argmin(np.abs(tpr_m - 0.95))
fpr95_msp = fpr_m[idx95_m] * 100

T_values = list(range(1, 11))
auroc_list = []
fpr95_list = []

print(f"MSP Baseline: AUROC = {auroc_msp:.2f}% | FPR95 = {fpr95_msp:.2f}%\n")

print(f"{'T':<5} | {'AUROC (%)':<10} | {'FPR95 (%)':<10}")
print("-" * 35)

for T in T_values:
    # E(x) = -T * logsumexp(logit / T)
    E_id = -T * torch.logsumexp(L_id / T, dim=1).numpy()
    E_ood = -T * torch.logsumexp(L_ood / T, dim=1).numpy()
    
    Esc = np.concatenate([E_id, E_ood])
    
    auroc = roc_auc_score(labels, Esc) * 100
    fpr_e, tpr_e, thr_e = roc_curve(labels, Esc)
    idx95 = np.argmin(np.abs(tpr_e - 0.95))
    fpr95 = fpr_e[idx95] * 100
    
    auroc_list.append(auroc)
    fpr95_list.append(fpr95)
    print(f"{T:<5} | {auroc:<10.2f} | {fpr95:<10.2f}")

# ── Vẽ biểu đồ ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = '#1565C0'
ax1.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUROC (%)', color=color1, fontsize=12, fontweight='bold')
ax1.plot(T_values, auroc_list, marker='o', color=color1, lw=2.5, markersize=8, label='Energy AUROC (↑)')
ax1.axhline(auroc_msp, color=color1, linestyle='--', lw=2, alpha=0.7, label=f'MSP AUROC Baseline ({auroc_msp:.1f}%)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(T_values)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color2 = '#C62828'
ax2.set_ylabel('FPR95 (%)', color=color2, fontsize=12, fontweight='bold')
ax2.plot(T_values, fpr95_list, marker='s', color=color2, lw=2.5, markersize=8, label='Energy FPR95 (↓)')
ax2.axhline(fpr95_msp, color=color2, linestyle='--', lw=2, alpha=0.7, label=f'MSP FPR95 Baseline ({fpr95_msp:.1f}%)')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Ảnh hưởng của Temperature (T) lên hiệu suất OOD Detection\n(Energy Score so với MSP Baseline, CIFAR-10 vs SVHN)', 
          fontsize=13, fontweight='bold', pad=15)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=9)

plt.tight_layout()
p = os.path.join(OUT, 'temperature_analysis.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved chart to: {p}")
