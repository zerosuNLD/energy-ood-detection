import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# ── SETUP ────────────────────────────────────────────────────────────
BASE = os.getcwd()
CKPT = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA = os.path.join(BASE, 'data')
OUT_DIR = os.path.join(BASE, 'results', 'charts_combined')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MU  = (0.4914, 0.4822, 0.4465)
SIG = (0.2023, 0.1994, 0.2010)

BLUE = '#1f77b4'
RED = '#d62728'
GREEN = '#2ca02c'
ORANGE = '#ff7f0e'

# ── MODEL DEFINITION ──────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.drop  = drop
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if in_ch != out_ch else None
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
        self.net = nn.Sequential(*[BasicBlock(in_ch if i==0 else out_ch, out_ch, stride if i==0 else 1, drop) for i in range(n)])
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

def main():
    print(f"Loading Model... Device: {DEVICE}")
    model = WideResNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    # ── DATA LOADING ──────────────────────────────────────────────────────
    tf_no_resize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MU, SIG)])
    tf_resize = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MU, SIG)])

    print("Loading datasets...")
    cifar10 = torchvision.datasets.CIFAR10(DATA, train=False, transform=tf_no_resize, download=False)
    svhn = torchvision.datasets.SVHN(DATA, split='test', transform=tf_no_resize, download=False)
    lsun = torchvision.datasets.ImageFolder(os.path.join(DATA, 'LSUN_resize'), transform=tf_resize)
    tiny = torchvision.datasets.ImageFolder(os.path.join(DATA, 'tiny-imagenet-200', 'test'), transform=tf_resize)

    loaders = {
        'CIFAR-10': torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=False, num_workers=4),
        'SVHN': torch.utils.data.DataLoader(svhn, batch_size=256, shuffle=False, num_workers=4),
        'LSUN': torch.utils.data.DataLoader(lsun, batch_size=256, shuffle=False, num_workers=4),
        'Tiny-ImageNet': torch.utils.data.DataLoader(tiny, batch_size=256, shuffle=False, num_workers=4),
    }

    @torch.no_grad()
    def get_logits(loader):
        L = []
        for x, y in loader:
            L.append(model(x.to(DEVICE)).cpu())
        return torch.cat(L)

    print("Computing logits...")
    logits_all = {name: get_logits(ldr) for name, ldr in loaders.items()}

    # ── METRIC CALCULATION ────────────────────────────────────────────────
    # OOD params
    ood_configs = {
        'SVHN': {'T': 0.5},
        'LSUN': {'T': 2.0},
        'Tiny-ImageNet': {'T': 1.0}
    }

    def calc_scores(logits, T):
        energy = -T * torch.logsumexp(logits / T, dim=1).numpy()
        msp = -torch.softmax(logits, dim=1).max(dim=1).values.numpy()
        return energy, msp

    L_id = logits_all['CIFAR-10']
    results = {}

    for ood_name in ['SVHN', 'LSUN', 'Tiny-ImageNet']:
        T = ood_configs[ood_name]['T']
        E_id, M_id = calc_scores(L_id, T)
        E_ood, M_ood = calc_scores(logits_all[ood_name], T)
        
        labels = np.concatenate([np.zeros(len(E_id)), np.ones(len(E_ood))])
        
        # Calculate Energy metrics
        s_E = np.concatenate([E_id, E_ood])
        auroc_E = roc_auc_score(labels, s_E) * 100
        fpr_E, tpr_E, thr_E = roc_curve(labels, s_E)
        idx95_E = np.argmin(np.abs(tpr_E - 0.95))
        fpr95_E = fpr_E[idx95_E] * 100
        gamma_E = thr_E[idx95_E]
        
        prec_E, rec_E, _ = precision_recall_curve(labels, s_E)
        f1_E = 2 * (prec_E * rec_E) / (prec_E + rec_E + 1e-8)
        best_f1_E = np.max(f1_E)
        aupr_E = auc(rec_E, prec_E) * 100
        
        # Calculate MSP metrics
        s_M = np.concatenate([M_id, M_ood])
        auroc_M = roc_auc_score(labels, s_M) * 100
        fpr_M, tpr_M, thr_M = roc_curve(labels, s_M)
        idx95_M = np.argmin(np.abs(tpr_M - 0.95))
        fpr95_M = fpr_M[idx95_M] * 100
        gamma_M = thr_M[idx95_M]
        
        prec_M, rec_M, _ = precision_recall_curve(labels, s_M)
        f1_M = 2 * (prec_M * rec_M) / (prec_M + rec_M + 1e-8)
        best_f1_M = np.max(f1_M)
        aupr_M = auc(rec_M, prec_M) * 100

        results[ood_name] = {
            'E_id': E_id, 'E_ood': E_ood, 'M_id': M_id, 'M_ood': M_ood,
            'gamma_E': gamma_E, 'fpr95_E': fpr95_E, 'auroc_E': auroc_E, 'best_f1_E': best_f1_E, 'aupr_E': aupr_E,
            'gamma_M': gamma_M, 'fpr95_M': fpr95_M, 'auroc_M': auroc_M, 'best_f1_M': best_f1_M, 'aupr_M': aupr_M,
            'fpr_arr_E': fpr_E, 'tpr_arr_E': tpr_E, 'thr_arr_E': thr_E,
            'prec_arr_E': prec_E, 'rec_arr_E': rec_E,
            'fpr_arr_M': fpr_M, 'tpr_arr_M': tpr_M, 'thr_arr_M': thr_M,
            'prec_arr_M': prec_M, 'rec_arr_M': rec_M,
            'labels': labels, 's_E': s_E, 's_M': s_M
        }

        # Print stats mapping
        print(f"=== {ood_name} (T={T}) ===")
        print(f"  Energy: FPR95={fpr95_E:.2f}%, AUROC={auroc_E:.2f}%, F1={best_f1_E:.4f}, AUPR={aupr_E/100:.4f}")
        print(f"  MSP   : FPR95={fpr95_M:.2f}%, AUROC={auroc_M:.2f}%, F1={best_f1_M:.4f}, AUPR={aupr_M/100:.4f}")
        print(f"  Energy ID   (mu, sigma): {np.mean(E_id):.3f}, {np.std(E_id):.3f}")
        print(f"  Energy OOD  (mu, sigma): {np.mean(E_ood):.3f}, {np.std(E_ood):.3f}")
        print(f"  Energy Delta mu: {np.mean(E_ood) - np.mean(E_id):.3f}")

    # ── PLOTTING ──────────────────────────────────────────────────────────

    # 1. Energy Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phân phối Energy Score: CIFAR-10 (ID) vs SVHN/LSUN/Tiny-ImageNet (OOD)", fontsize=16, fontweight='bold', y=1.02)
    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        T = ood_configs[ood_name]['T']
        ax.hist(r['E_id'], bins=100, alpha=0.6, color=BLUE, density=True, label=f'CIFAR-10 ID\n$\\mu$={np.mean(r["E_id"]):.2f}, $\\sigma$={np.std(r["E_id"]):.2f}')
        ax.hist(r['E_ood'], bins=100, alpha=0.6, color=RED, density=True, label=f'{ood_name} OOD\n$\\mu$={np.mean(r["E_ood"]):.2f}, $\\sigma$={np.std(r["E_ood"]):.2f}')
        ax.axvline(r['gamma_E'], color='black', lw=2, ls='--', label=f'Ngưỡng $\\gamma$ (FPR95)\n$\\Delta\\mu$={np.mean(r["E_ood"])-np.mean(r["E_id"]):.2f}')
        ax.set_xlabel(f'Energy Score (T={T})', fontsize=12)
        ax.set_ylabel('Mật độ xác suất', fontsize=12)
        ax.set_title(f'{ood_name} (T={T})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_combined_energy_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. MSP Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phân phối MSP Score: CIFAR-10 (ID) vs SVHN/LSUN/Tiny-ImageNet (OOD)", fontsize=16, fontweight='bold', y=1.02)
    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        ax.hist(r['M_id'], bins=100, alpha=0.6, color=BLUE, density=True, label=f'CIFAR-10 ID\n$\\mu$={np.mean(r["M_id"]):.2f}, $\\sigma$={np.std(r["M_id"]):.2f}')
        ax.hist(r['M_ood'], bins=100, alpha=0.6, color=RED, density=True, label=f'{ood_name} OOD\n$\\mu$={np.mean(r["M_ood"]):.2f}, $\\sigma$={np.std(r["M_ood"]):.2f}')
        ax.axvline(r['gamma_M'], color='black', lw=2, ls='--', label=f'Ngưỡng $\\gamma$ (FPR95)\n$\\Delta\\mu$={np.mean(r["M_ood"])-np.mean(r["M_id"]):.2f}')
        ax.set_xlabel('MSP Score', fontsize=12)
        ax.set_ylabel('Mật độ xác suất', fontsize=12)
        ax.set_title(f'{ood_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_combined_msp_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. ROC Curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ROC Curve — Energy Score vs MSP", fontsize=16, fontweight='bold', y=1.02)
    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        T = ood_configs[ood_name]['T']
        ax.plot(r['fpr_arr_E'], r['tpr_arr_E'], color=BLUE, lw=2, label=f'Energy (T={T}) | AUROC={r["auroc_E"]:.2f}%')
        ax.plot(r['fpr_arr_M'], r['tpr_arr_M'], color=RED, lw=2, label=f'MSP Baseline | AUROC={r["auroc_M"]:.2f}%')
        ax.plot([r['fpr95_E']/100], [0.95], marker='o', color=BLUE, markersize=8)
        ax.plot([r['fpr95_M']/100], [0.95], marker='o', color=RED, markersize=8)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title(f'{ood_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_combined_roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Precision-Recall Curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Precision-Recall Curve — Energy Score vs MSP", fontsize=16, fontweight='bold', y=1.02)
    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        T = ood_configs[ood_name]['T']
        ax.plot(r['rec_arr_E'], r['prec_arr_E'], color=BLUE, lw=2, label=f'Energy (T={T}) | AUPR={r["aupr_E"]/100:.4f}')
        ax.plot(r['rec_arr_M'], r['prec_arr_M'], color=RED, lw=2, label=f'MSP Baseline | AUPR={r["aupr_M"]/100:.4f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.3, 1.05])
        ax.set_xlabel('Recall (TPR)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{ood_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_combined_pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. FPR & TPR vs Threshold
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FPR và TPR theo ngưỡng $\\gamma$ — Energy Score", fontsize=16, fontweight='bold', y=1.02)
    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        T = ood_configs[ood_name]['T']
        # Limit threshold range to where TPR is between 1 and 0 (valid range)
        valid_idx = (r['thr_arr_E'] >= np.min(r['s_E'])) & (r['thr_arr_E'] <= np.max(r['s_E']))
        thr = r['thr_arr_E'][valid_idx]
        tpr = r['tpr_arr_E'][valid_idx]
        fpr = r['fpr_arr_E'][valid_idx]

        ax.plot(thr, tpr, label='TPR (Recall)', color=GREEN, lw=2)
        ax.plot(thr, fpr, label='FPR (Báo nhầm)', color=RED, lw=2)
        ax.axvline(r['gamma_E'], color='black', ls='--', lw=1.5, label=f'Ngưỡng $\\gamma$ tại TPR=95% ({r["gamma_E"]:.2f})')
        ax.axhline(0.95, color='gray', ls=':', lw=1.5)
        
        ax.set_xlim([np.percentile(r['s_E'], 1), np.percentile(r['s_E'], 99)])
        ax.set_xlabel(f'Ngưỡng phân loại $\\gamma$ (T={T})', fontsize=12)
        ax.set_ylabel('Tỷ lệ (%)', fontsize=12)
        ax.set_title(f'{ood_name} (T={T})', fontsize=14, fontweight='bold')
        ax.legend(loc='center right', fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_combined_fpr_tpr.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Bar Chart Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sánh tổng hợp 4 chỉ số: Energy Score vs MSP", fontsize=16, fontweight='bold', y=1.02)
    metrics = ['FPR95↓', 'AUROC↑', 'Best F1↑', 'AUPR↑']
    x = np.arange(len(metrics))
    width = 0.35

    for ax, ood_name in zip(axes, ['SVHN', 'LSUN', 'Tiny-ImageNet']):
        r = results[ood_name]
        T = ood_configs[ood_name]['T']
        
        energy_vals = [r['fpr95_E'], r['auroc_E'], r['best_f1_E']*100, r['aupr_E']]
        msp_vals = [r['fpr95_M'], r['auroc_M'], r['best_f1_M']*100, r['aupr_M']]
        
        rects1 = ax.bar(x - width/2, energy_vals, width, label=f'Energy (T={T})', color=BLUE)
        rects2 = ax.bar(x + width/2, msp_vals, width, label='MSP Baseline', color=RED)
        
        ax.set_ylabel('Giá trị (%)', fontsize=12)
        ax.set_title(f'{ood_name} (T={T})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 115])
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        autolabel(rects1)
        autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_combined_bar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\\nAll 6 combined figures saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
