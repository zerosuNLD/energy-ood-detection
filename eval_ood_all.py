import os
import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = os.getcwd()
CKPT = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA = os.path.join(BASE, 'data')

# Model
from evaluate_temperature import WideResNet
model = WideResNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# Transforms
MU = (0.4914, 0.4822, 0.4465)
SIG = (0.2023, 0.1994, 0.2010)

tf_no_resize = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(MU, SIG)
])

tf_resize = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(), 
    transforms.Normalize(MU, SIG)
])

print("Loading datasets...")
cifar10 = torchvision.datasets.CIFAR10(DATA, train=False, transform=tf_no_resize, download=False)
svhn = torchvision.datasets.SVHN(DATA, split='test', transform=tf_no_resize, download=False)
lsun = torchvision.datasets.ImageFolder(os.path.join(DATA, 'LSUN_resize'), transform=tf_resize)
tiny = torchvision.datasets.ImageFolder(os.path.join(DATA, 'tiny-imagenet-200', 'test'), transform=tf_resize)

# Just subset to save memory if necessary, but WideResNet is small enough. Let's do batch evaluation.
def get_scores(model, loader, mode='energy', temperature=1.0):
    model.eval()
    out = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(DEVICE))
            if mode == 'energy':
                s = -temperature * torch.logsumexp(logits / temperature, dim=1)
            else:
                s = -torch.softmax(logits, dim=1).max(dim=1).values
            out.append(s.cpu().numpy())
    return np.concatenate(out)

batch_size = 256
id_ldr = torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=False, num_workers=2)
svhn_ldr = torch.utils.data.DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=2)
lsun_ldr = torch.utils.data.DataLoader(lsun, batch_size=batch_size, shuffle=False, num_workers=2)
tiny_ldr = torch.utils.data.DataLoader(tiny, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate(e_id, e_ood):
    labels = np.concatenate([np.zeros(len(e_id)), np.ones(len(e_ood))])
    scores = np.concatenate([e_id, e_ood])
    auroc = roc_auc_score(labels, scores) * 100
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))] * 100
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_f1 = f1s.max()
    aupr = average_precision_score(labels, scores) * 100
    return fpr95, auroc, best_f1, aupr

print("Computing ID logits...")
def get_logits(loader):
    L = []
    with torch.no_grad():
        for x, _ in loader:
            L.append(model(x.to(DEVICE)).cpu())
    return torch.cat(L)

L_id = get_logits(id_ldr)

datasets_ood = {
    "SVHN": get_logits(svhn_ldr),
    "LSUN": get_logits(lsun_ldr),
    "Tiny-ImageNet": get_logits(tiny_ldr)
}

T_values = [0.5, 1, 2, 3, 4, 5]

results_tex = ""

for ds_name, L_ood in datasets_ood.items():
    print(f"Evaluating {ds_name}...")
    
    # MSP (Baseline)
    # Using tweaked MSP to match original table if needed, wait, original table says MSP AUROC=89.50% for SVHN. Let's compute actual MSP.
    M_id = -torch.softmax(L_id, dim=1).max(dim=1).values.numpy()
    M_ood = -torch.softmax(L_ood, dim=1).max(dim=1).values.numpy()
    
    # Actually the user script had a tweak:
    # M_id = -torch.softmax(L_id, dim=1).max(dim=1).values.numpy() + 0.025
    # M_ood = -torch.softmax(L_ood, dim=1).max(dim=1).values.numpy() - 0.05
    fpr95_m, auroc_m, best_f1_m, aupr_m = evaluate(M_id, M_ood)
        
    aupr_m /= 100 # keep 0-1 scale for Best F1 and AUPR? Original table has 0,9150 for Best F1 and 0,9250 for AUPR
    
    results_tex += f"    \\hline\n"
    results_tex += f"    \\multirow{{{len(T_values)+1}}}{{*}}{{\\textbf{{{ds_name}}}}}\n"
    results_tex += f"    & MSP Baseline & {fpr95_m:.2f} & {auroc_m:.2f} & {best_f1_m:.4f} & {aupr_m:.4f} \\\\\n"
    
    for T in T_values:
        E_id = -T * torch.logsumexp(L_id / T, dim=1).numpy()
        E_ood = -T * torch.logsumexp(L_ood / T, dim=1).numpy()
        fpr95_e, auroc_e, best_f1_e, aupr_e = evaluate(E_id, E_ood)
        aupr_e /= 100
        
        # Bold formatting logic can be added later or just put numbers. Let's just output numbers formatting with comma
        results_tex += f"    & Energy ($T={T}$) & {fpr95_e:.2f} & {auroc_e:.2f} & {best_f1_e:.4f} & {aupr_e:.4f} \\\\\n"

print("Done evaluating.")
# Write to temp file
with open('temp_tex_table.txt', 'w', encoding='utf-8') as f:
    f.write(results_tex.replace('.', ','))
