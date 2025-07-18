#!/usr/bin/env python
# eval_robustness_plus.py  (save log + plot)
# ===============================================================
# Clean + 8 corruptions × 3 severities  (25 sets)
# 保存结果到 CSV 与 PNG 条形图
# ===============================================================
import argparse, datetime, random, csv, os
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

# ───────── reproducibility ─────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED); random.seed(SEED)

# ───────── 1. corruption wrappers (可 pickle) ─────────
class GaussianNoise:
    def __init__(self, std): self.std = std
    def __call__(self, x): return torch.clamp(x + torch.randn_like(x)*self.std, 0, 1)

class MotionBlur:
    def __init__(self, k):
        kern = torch.zeros((k, k)); kern[k//2] = 1.0/k
        self.kern = kern.unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)   # (3,1,k,k)
    def __call__(self, x):
        return torch.nn.functional.conv2d(x.unsqueeze(0), self.kern.to(x),
                                          padding="same", groups=3)[0]

class BrightnessShift:
    def __init__(self, d): self.d = d
    def __call__(self, x): return torch.clamp(x + self.d, 0, 1)

class OcclusionPatch:
    def __init__(self, r): self.r = r
    def __call__(self, x):
        h,w = x.shape[1:]; th,tw = int(h*self.r), int(w*self.r)
        sx,sy = random.randint(0,h-th), random.randint(0,w-tw)
        x = x.clone(); x[:,sx:sx+th,sy:sy+tw] = 0; return x

class JPEG:
    def __init__(self, q): self.q=q; self.to_pil=transforms.ToPILImage(); self.to_tensor=transforms.ToTensor()
    def __call__(self, x):
        buf = BytesIO(); self.to_pil(x).save(buf, format="JPEG", quality=self.q); buf.seek(0)
        return self.to_tensor(Image.open(buf))

class SaltPepper:
    """给每个像素独立加盐椒噪声：p/2 概率设 0，p/2 概率设 1"""
    def __init__(self, p: float):
        assert 0. < p < 1.
        self.p = p
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(x)                 # 与 x 同形状 [3,H,W]
        x = x.clone()
        x[mask < self.p/2]  = 0.
        x[(mask >= self.p/2) & (mask < self.p)] = 1.
        return x

class RandAffine:
    def __init__(self, deg):
        self.t = transforms.RandomAffine(
            degrees=deg, translate=(.1, .1)
        )
    def __call__(self, x):
        x = self.t(x)
        # 强制回到 224×224，防止极端版本问题
        if x.shape[1:] != (224, 224):
            x = F.resize(x, [224, 224])
        return x

class RandScale:
    """
    随机缩放后，再 resize 回 224×224，保证 batch 内一致尺寸。
    """
    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s  = random.uniform(self.lo, self.hi)
        nh = int(round(x.shape[1] * s))
        nw = int(round(x.shape[2] * s))
        # 1. 缩放到新尺寸
        x = F.resize(x, [nh, nw])
        # 2. 再缩放/填充回 224×224（保持比例扭曲很小，可接受）
        return F.resize(x, [224, 224])

# (tag, label, transform) triples
ROBUST_TESTS = [
    *(("gauss", f"σ={s}", GaussianNoise(s))       for s in (0.05,0.10,0.20)),
    *(("mblur", f"k={k}", MotionBlur(k))          for k in (7,11,15)),
    *(("bright",f"Δ={d}", BrightnessShift(d))     for d in (0.2,0.4,0.6)),
    *(("occ",   f"r={r}", OcclusionPatch(r))      for r in (0.2,0.3,0.4)),
    *(("jpeg",  f"q={q}", JPEG(q))                for q in (50,30,10)),
    *(("sp",    f"p={p}", SaltPepper(p))          for p in (0.02,0.04,0.08)),
    *(("aff",   f"deg={d}", RandAffine(d))        for d in (15,25,45)),
    *(("scale", f"{lo}-{hi}", RandScale(lo,hi))   for lo,hi in ((0.7,1.3),(0.6,1.4),(0.5,1.5))),
]

# ───────── 2. CLI ─────────
argp = argparse.ArgumentParser()
argp.add_argument("--ckpt", type=Path, required=True)
argp.add_argument("--data_dir", type=Path, default=Path("dataset"))
argp.add_argument("--batch", type=int, default=32)
argp.add_argument("--workers", type=int, default=4)
argp.add_argument("--tta", action="store_true")
argp.add_argument("--out_dir", type=Path, default=Path("outputs_robustorg"))
cfg = argp.parse_args()

# ───────── 3. helpers ─────────
def build_loader(transform):
    return DataLoader(
        datasets.ImageFolder(cfg.data_dir/"test", transform),
        cfg.batch, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)

base_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def build_model(nc):
    m = timm.create_model("convnext_tiny.fb_in22k", pretrained=False)
    m.head.fc = nn.Linear(m.head.fc.in_features, nc)
    sd = torch.load(cfg.ckpt, map_location="cpu")["state"]
    m.load_state_dict(sd, strict=True)
    return m.to(DEVICE).eval()

@torch.no_grad()
def eval_loader(ld):
    correct = cnt = 0
    for x,y in ld:
        y=y.to(DEVICE)
        out = flip_tta(x) if cfg.tta else model(x.to(DEVICE))
        correct += out.argmax(1).eq(y).sum().item()
        cnt     += y.size(0)
    return correct/cnt*100

# ───────── 4. main ─────────
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.out_dir.mkdir(exist_ok=True)
    csv_path = cfg.out_dir/f"robust_results_{ts}.csv"
    png_path = cfg.out_dir/f"robust_bar_{ts}.png"

    clean_ld = build_loader(base_tf)
    global model; model = build_model(len(clean_ld.dataset.classes))

    global flip_tta
    @torch.no_grad()
    def flip_tta(x):
        xf = torch.flip(x,dims=[3])
        o  = model(torch.cat([x,xf],0).to(DEVICE)).softmax(1)
        return o[:x.size(0)]+o[x.size(0):]

    # 4-1 评测
    results = [("clean","-", eval_loader(clean_ld))]
    for tag,info,fn in ROBUST_TESTS:
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), fn,
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        acc = eval_loader(build_loader(tf))
        results.append((tag,info,acc))
        print(f"{tag:<6} {info:<8} | Acc {acc:.2f}%")

    # 4-2 保存 CSV
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Type","Param","Accuracy"])
        w.writerows(results)
    print(f"\n📄 结果已保存: {csv_path}")

    # 4-3 保存条形图
    try:
        import matplotlib.pyplot as plt
        labels = [f"{t}:{i}" for t,i,_ in results]
        accs   = [a for *_,a in results]
        plt.figure(figsize=(12,4))
        plt.bar(range(len(accs)), accs)
        plt.xticks(range(len(accs)), labels, rotation=90)
        plt.ylabel("Accuracy (%)"); plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"🖼  条形图已保存: {png_path}")
    except ImportError:
        print("⚠ 未安装 matplotlib，跳过绘图")

# ───────── 5. entry ─────────
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
