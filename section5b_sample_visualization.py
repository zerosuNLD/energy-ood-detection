
# ================================================================
# SECTION 5b – Hiển thị 1 mẫu ID và 1 mẫu OOD với Energy Score
# ================================================================
# Mục đích: minh hoạ trực quan xem đâu là In-Distribution (ID)
# và đâu là Out-of-Distribution (OOD) dựa trên energy score.
#
# Quy ước:
#   Energy THẤP  →  In-Distribution  (mô hình quen thuộc)
#   Energy CAO   →  Out-of-Distribution  (mô hình xa lạ)
# ================================================================

CIFAR10_CLASSES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ── Lấy 1 ảnh từ CIFAR-10 (ID) ──────────────────────────────────
cifar_iter  = iter(id_ldr)
id_imgs, id_labels = next(cifar_iter)        # lấy batch đầu tiên
id_img   = id_imgs[0:1].to(DEVICE)           # 1 ảnh duy nhất
id_label = id_labels[0].item()

# ── Lấy 1 ảnh từ SVHN (OOD) ──────────────────────────────────────
svhn_iter   = iter(ood_ldr)
ood_imgs, _ = next(svhn_iter)
ood_img     = ood_imgs[0:1].to(DEVICE)       # 1 ảnh duy nhất

# ── Tính Energy Score cho từng ảnh ───────────────────────────────
model.eval()
with torch.no_grad():
    logits_id  = model(id_img)
    logits_ood = model(ood_img)

energy_id  = (-T * torch.logsumexp(logits_id  / T, dim=1)).item()
energy_ood = (-T * torch.logsumexp(logits_ood / T, dim=1)).item()

# Softmax prob để hiện top-3
prob_id  = torch.softmax(logits_id,  dim=1)[0].cpu().numpy()
prob_ood = torch.softmax(logits_ood, dim=1)[0].cpu().numpy()

top3_id  = np.argsort(prob_id)[::-1][:3]
top3_ood = np.argsort(prob_ood)[::-1][:3]

# ── Giải chuẩn hoá ảnh để plot ───────────────────────────────────
def denormalize(tensor):
    """Đảo Normalize CIFAR-10 để hiển thị ảnh đúng màu."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR_STD).view(3, 1, 1)
    img  = tensor.squeeze(0).cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()

img_id_show  = denormalize(id_img.cpu())
img_ood_show = denormalize(ood_img.cpu())

# ── Vẽ hình ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Minh hoạ: Phân loại In-Distribution vs Out-of-Distribution\n'
    'bằng Energy Score  |  Energy THẤP = ID   |   Energy CAO = OOD',
    fontsize=13, fontweight='bold'
)

# --- In-Distribution sample ---
ax = axes[0]
ax.imshow(img_id_show)
ax.axis('off')
verdict = 'IN-DISTRIBUTION ✓'
color   = '#1B5E20'
ax.set_title(
    f'Tập CIFAR-10  →  Nhãn thật: {CIFAR10_CLASSES[id_label]}\n'
    f'Energy Score : {energy_id:.3f}\n'
    f'Phán quyết   : {verdict}',
    fontsize=11, color=color, fontweight='bold', pad=8
)
for rank, cls_idx in enumerate(top3_id):
    ax.text(
        0.03, 0.05 + rank * 0.10,
        f'{CIFAR10_CLASSES[cls_idx]}: {prob_id[cls_idx]*100:.1f}%',
        transform=ax.transAxes, fontsize=9,
        color='white', fontweight='bold',
        bbox=dict(facecolor='#1976D2', alpha=0.75, boxstyle='round,pad=0.2')
    )

# --- OOD sample ---
ax = axes[1]
ax.imshow(img_ood_show)
ax.axis('off')
verdict_ood = 'OUT-OF-DISTRIBUTION ✗'
color_ood   = '#B71C1C'
ax.set_title(
    f'Tập SVHN  →  Domain: Street View Numbers\n'
    f'Energy Score : {energy_ood:.3f}\n'
    f'Phán quyết   : {verdict_ood}',
    fontsize=11, color=color_ood, fontweight='bold', pad=8
)
for rank, cls_idx in enumerate(top3_ood):
    ax.text(
        0.03, 0.05 + rank * 0.10,
        f'{CIFAR10_CLASSES[cls_idx]}: {prob_ood[cls_idx]*100:.1f}%',
        transform=ax.transAxes, fontsize=9,
        color='white', fontweight='bold',
        bbox=dict(facecolor='#D32F2F', alpha=0.75, boxstyle='round,pad=0.2')
    )

# ── Nhận xét terminal ────────────────────────────────────────────
print('=' * 60)
print('  KẾT QUẢ PHÂN LOẠI MẪU ĐƠN LẺ')
print('=' * 60)
print(f'  [ID  – CIFAR-10]  Energy = {energy_id:+.3f}')
print(f'     Top-3: ' + ', '.join(
    f'{CIFAR10_CLASSES[i]}({prob_id[i]*100:.1f}%)' for i in top3_id))
print()
print(f'  [OOD – SVHN    ]  Energy = {energy_ood:+.3f}')
print(f'     Top-3: ' + ', '.join(
    f'{CIFAR10_CLASSES[i]}({prob_ood[i]*100:.1f}%)' for i in top3_ood))
print()
delta = energy_ood - energy_id
status = "OOD energy CAO HƠN → tách biệt đúng ✓" if delta > 0 else "OOD energy THẤP HƠN → cần xem lại ✗"
print(f'  ΔEnergy (OOD − ID) = {delta:+.3f}  →  {status}')
print('=' * 60)

plt.tight_layout()
plt.savefig('sample_id_vs_ood.png', dpi=150, bbox_inches='tight')
plt.show()
print('Đã lưu: sample_id_vs_ood.png')
