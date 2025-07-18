#!/usr/bin/env python
# eval_robustness_train.py
# ===============================================================
# Robustness eval for ConvNeXt-Tiny (MixUp+SWA) trained by train.py
# Clean + 8 corruptions × 3 severities → CSV & PNG
# ===============================================================
import argparse, datetime, random, csv
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F, InterpolationMode
from tqdm import tqdm

# reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED); random.seed(SEED)

# ───────── 可-pickle 扰动封装 ─────────
class GaussianNoise:
    def __init__(self, std): self.std=std
    def __call__(self,x): return torch.clamp(x+torch.randn_like(x)*self.std,0,1)

class MotionBlur:
    def __init__(self,k):
        kern=torch.zeros((k,k)); kern[k//2]=1/k
        self.kern=kern.unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
    def __call__(self,x):
        return torch.nn.functional.conv2d(x.unsqueeze(0), self.kern.to(x),
                                          padding='same', groups=3)[0]

class BrightnessShift:
    def __init__(self,d): self.d=d
    def __call__(self,x): return torch.clamp(x+self.d,0,1)

class OcclusionPatch:
    def __init__(self,r): self.r=r
    def __call__(self,x):
        h,w=x.shape[1:]; th,tw=int(h*self.r),int(w*self.r)
        sx,sy=random.randint(0,h-th),random.randint(0,w-tw)
        x=x.clone(); x[:,sx:sx+th,sy:sy+tw]=0; return x

class JPEG:
    def __init__(self,q):
        self.q=q; self.to_pil=transforms.ToPILImage(); self.to_tensor=transforms.ToTensor()
    def __call__(self,x):
        buf=BytesIO(); self.to_pil(x).save(buf,format='JPEG',quality=self.q); buf.seek(0)
        return self.to_tensor(Image.open(buf))

class SaltPepper:
    def __init__(self,p): self.p=p
    def __call__(self,x):
        mask=torch.rand_like(x)
        x=x.clone()
        x[mask<self.p/2]=0; x[(mask>=self.p/2)&(mask<self.p)]=1
        return x

class RandAffine:
    def __init__(self,deg):
        self.t=transforms.RandomAffine(deg,translate=(.1,.1),
                                       interpolation=InterpolationMode.BILINEAR)
    def __call__(self,x):
        return self.t(x)

class RandScale:
    def __init__(self,lo,hi): self.lo,self.hi=lo,hi
    def __call__(self,x):
        s=random.uniform(self.lo,self.hi)
        nh,nw=int(x.shape[1]*s),int(x.shape[2]*s)
        x=F.resize(x,[nh,nw],antialias=True)
        return F.resize(x,[224,224],antialias=True)

ROBUST = [
    *(("gauss",f"σ={s}",GaussianNoise(s))           for s in (0.05,0.10,0.20)),
    *(("mblur",f"k={k}",MotionBlur(k))              for k in (7,11,15)),
    *(("bright",f"Δ={d}",BrightnessShift(d))        for d in (0.2,0.4,0.6)),
    *(("occ",f"r={r}",OcclusionPatch(r))            for r in (0.2,0.3,0.4)),
    *(("jpeg",f"q={q}",JPEG(q))                     for q in (50,30,10)),
    *(("sp",f"p={p}",SaltPepper(p))                 for p in (0.02,0.04,0.08)),
    *(("aff",f"deg={d}",RandAffine(d))              for d in (15,25,45)),
    *(("scale",f"{lo}-{hi}",RandScale(lo,hi))       for lo,hi in ((0.7,1.3),(0.6,1.4),(0.5,1.5))),
]

# ───────── CLI ─────────
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt",type=Path,required=True,
                help="权重文件，如 checkpoints/swa_best.pth")
ap.add_argument("--data_dir",type=Path,default=Path("dataset"))
ap.add_argument("--batch",type=int,default=32)
ap.add_argument("--workers",type=int,default=4)
ap.add_argument("--tta",action="store_true",help="水平翻转 TTA")
ap.add_argument("--out_dir",type=Path,default=Path("output_org"))
cfg=ap.parse_args()

# ───────── Data loader ─────────
base_tf=transforms.Compose([
    transforms.Resize(256,antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
def loader(tf): return DataLoader(
    datasets.ImageFolder(cfg.data_dir/"test",tf),
    cfg.batch,shuffle=False,num_workers=cfg.workers,pin_memory=True)

# ───────── Model ─────────
def build_model(num_cls):
    # 1) 构造空模型
    m = timm.create_model("convnext_tiny.fb_in22k", pretrained=False)
    m.head.fc = nn.Linear(m.head.fc.in_features, num_cls)

    # 2) 读取 checkpoint
    raw = torch.load(cfg.ckpt, map_location="cpu")
    sd  = raw.get("state_dict") or raw.get("state") or raw  # 兼容不同字段名

    # 3) 处理 DataParallel & SWA 前缀
    new_sd = {}
    for k, v in sd.items():
        if k == "n_averaged":          # SWA 统计量，模型用不到
            continue
        if k.startswith("module."):    # 去掉 DataParallel 前缀
            k = k[7:]
        new_sd[k] = v

    # 4) 加载
    m.load_state_dict(new_sd, strict=True)
    return m.to(DEVICE).eval()

# ───────── Eval util ─────────
@torch.no_grad()
def evaluate(ld):
    tot=correct=0
    for x,y in ld:
        y=y.to(DEVICE)
        out=flip_tta(x) if cfg.tta else model(x.to(DEVICE))
        correct+=out.argmax(1).eq(y).sum().item(); tot+=y.size(0)
    return correct/tot*100

# ───────── main ─────────
def main():
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.out_dir.mkdir(exist_ok=True)
    csv_path=cfg.out_dir/f"robust_results_{ts}.csv"
    png_path=cfg.out_dir/f"robust_bar_{ts}.png"

    clean_ld=loader(base_tf)
    global model; model=build_model(len(clean_ld.dataset.classes))

    global flip_tta
    @torch.no_grad()
    def flip_tta(x):
        xf=torch.flip(x,dims=[3])
        out=model(torch.cat([x,xf],0).to(DEVICE)).softmax(1)
        return out[:x.size(0)]+out[x.size(0):]

    # --- Run tests ---
    results=[("clean","-",evaluate(clean_ld))]
    print(f"🟢 clean        | Acc {results[0][2]:.2f}%")
    for tag,info,tf in ROBUST:
        acc=evaluate(loader(transforms.Compose([
            transforms.Resize(256,antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(), tf,
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])))
        results.append((tag,info,acc))
        print(f"🟠 {tag:<6} {info:<7}| Acc {acc:.2f}%")

    # --- Save CSV ---
    with open(csv_path,"w",newline="") as f:
        csv.writer(f).writerows([("Type","Param","Acc%")]+results)
    print(f"\n📄 CSV saved → {csv_path}")

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt
        labels=[f"{t}:{i}" for t,i,_ in results]
        accs=[a for *_,a in results]
        plt.figure(figsize=(13,4)); plt.bar(range(len(accs)),accs)
        plt.xticks(range(len(accs)),labels,rotation=90)
        plt.ylabel("Accuracy (%)"); plt.tight_layout()
        plt.savefig(png_path,dpi=150); plt.close()
        print(f"🖼  PNG saved → {png_path}")
    except ImportError:
        print("⚠ matplotlib 未安装，跳过绘图")

# ───────── entry ─────────
if __name__=="__main__":
    torch.multiprocessing.set_start_method("spawn",force=True)
    main()
