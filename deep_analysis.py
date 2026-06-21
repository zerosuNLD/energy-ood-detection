import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Reuse the model definition from the workspace
from evaluate_temperature import WideResNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = os.getcwd()
CKPT = os.path.join(BASE, 'wrn28_10_cifar10.pth')
DATA = os.path.join(BASE, 'data')
OUT_DIR = os.path.join(BASE, 'results', 'charts_combined')
os.makedirs(OUT_DIR, exist_ok=True)

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
tiny = torchvision.datasets.ImageFolder(os.path.join(DATA, 'tiny-imagenet-200', 'test'), transform=tf_resize)

id_ldr = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=False)
svhn_ldr = torch.utils.data.DataLoader(svhn, batch_size=256, shuffle=False)
tiny_ldr = torch.utils.data.DataLoader(tiny, batch_size=256, shuffle=False)

print("Loading model...")
model = WideResNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# Extract features
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        x = self.model.conv0(x)
        x = self.model.block1(x)
        x = self.model.block2(x)
        x = self.model.block3(x)
        x = F.relu(self.model.bn(x), inplace=True)
        feat = F.adaptive_avg_pool2d(x, 1).view(-1, self.model.out_ch)
        logits = self.model.fc(feat)
        return feat, logits

feat_model = FeatureExtractor(model).to(DEVICE)
feat_model.eval()

def get_features_and_logits(loader, max_samples=2000):
    feats, logits = [], []
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            f, l = feat_model(x.to(DEVICE))
            feats.append(f.cpu().numpy())
            logits.append(l.cpu().numpy())
            count += len(x)
            if count >= max_samples:
                break
    return np.concatenate(feats)[:max_samples], np.concatenate(logits)[:max_samples]

print("Extracting features and logits...")
f_id, l_id = get_features_and_logits(id_ldr, 2000)
f_svhn, l_svhn = get_features_and_logits(svhn_ldr, 2000)
f_tiny, l_tiny = get_features_and_logits(tiny_ldr, 2000)

print("Computing maximum logits distributions...")
max_l_id = np.max(l_id, axis=1)
max_l_svhn = np.max(l_svhn, axis=1)
max_l_tiny = np.max(l_tiny, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(max_l_id, bins=50, alpha=0.5, density=True, label='CIFAR-10 (ID)')
plt.hist(max_l_svhn, bins=50, alpha=0.5, density=True, label='SVHN (OOD)')
plt.hist(max_l_tiny, bins=50, alpha=0.5, density=True, label='Tiny-ImageNet (OOD)')
plt.title("Phân phối Maximum Logit")
plt.xlabel("Max Logit")
plt.ylabel("Mật độ")
plt.legend()
path_logits = os.path.join(OUT_DIR, 'fig7_max_logit_dist.png')
plt.savefig(path_logits, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {path_logits}")

print("Computing t-SNE for CIFAR-10 vs SVHN...")
tsne = TSNE(n_components=2, random_state=42)
all_f_svhn = np.vstack([f_id, f_svhn])
tsne_svhn = tsne.fit_transform(all_f_svhn)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_svhn[:2000, 0], tsne_svhn[:2000, 1], alpha=0.5, label='CIFAR-10 (ID)', s=10, c='blue')
plt.scatter(tsne_svhn[2000:, 0], tsne_svhn[2000:, 1], alpha=0.5, label='SVHN (OOD)', s=10, c='red')
plt.title("Trực quan hóa t-SNE Embedding: CIFAR-10 vs SVHN\nSVHN tạo thành các cụm riêng nhưng phân tán mạnh và có một số điểm xâm nhập sâu vào không gian đặc trưng của ID.")
plt.legend()
path_tsne_svhn = os.path.join(OUT_DIR, 'fig8_tsne_svhn.png')
plt.savefig(path_tsne_svhn, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {path_tsne_svhn}")

print("Computing t-SNE for CIFAR-10 vs Tiny-ImageNet...")
tsne_tiny = TSNE(n_components=2, random_state=42)
all_f_tiny = np.vstack([f_id, f_tiny])
tsne_res_tiny = tsne_tiny.fit_transform(all_f_tiny)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_res_tiny[:2000, 0], tsne_res_tiny[:2000, 1], alpha=0.5, label='CIFAR-10 (ID)', s=10, c='blue')
plt.scatter(tsne_res_tiny[2000:, 0], tsne_res_tiny[2000:, 1], alpha=0.5, label='Tiny-ImageNet (OOD)', s=10, c='orange')
plt.title("Trực quan hóa t-SNE Embedding: CIFAR-10 vs Tiny-ImageNet\nTiny-ImageNet chồng lấn mạnh mẽ với không gian đặc trưng của CIFAR-10 do tương đồng về ngữ nghĩa.")
plt.legend()
path_tsne_tiny = os.path.join(OUT_DIR, 'fig9_tsne_tiny.png')
plt.savefig(path_tsne_tiny, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {path_tsne_tiny}")
print("Done.")
