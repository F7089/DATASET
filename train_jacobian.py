# train_jacobian.py  – herbal_medicine/
# ===============================================================
# ConvNeXt-Tiny (IN-22k) + RandAugment + MixUp/CutMix (timm 0.9+)
# Jacobian Reg (λ=0.05, last 20 ep, 每4 batch) + EMA + SWA + AMP
# 时间戳输出；Loss & Acc 分图；Windows-safe (spawn)
# ===============================================================
import csv, random, datetime
from pathlib import Path
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from timm.utils import ModelEmaV2
from torchvision import transforms, datasets
from torchvision.transforms import (RandAugment, ColorJitter,
                                    RandomApply, GaussianBlur, RandomErasing)
from tqdm import tqdm
import matplotlib.pyplot as plt

# ───────── Hyper params ─────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "convnext_tiny.fb_in22k"
BATCH_SIZE = 32
EPOCHS, WARMUP_EPOCH = 120, 5
FREEZE_EPOCH, SWA_START = 10, 90          # 末30epoch做SWA
MIXUP_ALPHA, CUTMIX_ALPHA = 0.4, 1.0
JR_LAMBDA, JR_EVERY = 0.05, 4
HEAD_LR, BODY_LR, WD = 3e-4, 3e-5, 1e-4

torch.manual_seed(SEED); random.seed(SEED)

# ───────── Helper classes ─────────
class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self,v,n=1): self.val=v; self.sum+=v*n; self.count+=n; self.avg=self.sum/self.count
class EarlyStop:
    def __init__(self,p=20): self.p=p; self.best=-1; self.c=0
    def __call__(self,val): self.c=0 if val>self.best else self.c+1; self.best=max(self.best,val); return self.c>=self.p

# ───────── main ─────────
def main():
    # == 路径 ==
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT/"dataset"
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = ROOT/"outputs"/run_id
    CKPT, LOGS, FIGS = [OUT/p for p in ("ckpt","logs","figs")]
    for d in (CKPT,LOGS,FIGS): d.mkdir(parents=True, exist_ok=True)

    # == DataLoader ==
    train_tf = transforms.Compose([
        RandAugment(3,15),
        ColorJitter(0.4,0.4,0.4,0.1),
        RandomApply([GaussianBlur(3)],p=0.3),
        transforms.RandomResizedCrop(224,(0.7,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        RandomErasing(p=0.25,scale=(0.02,0.2)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_loader = DataLoader(
        datasets.ImageFolder(DATA/"train",train_tf),
        BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)           # 保证偶数 batch
    val_loader   = DataLoader(
        datasets.ImageFolder(DATA/"val",  eval_tf),
        BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(
        datasets.ImageFolder(DATA/"test", eval_tf),
        BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    NUM_CLASSES = len(train_loader.dataset.classes)

    # == Model ==
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # 冻结骨干
    for n,p in model.named_parameters():
        if "head.fc" not in n: p.requires_grad_(False)

    head_p = [p for n,p in model.named_parameters() if "head.fc" in n]
    body_p = [p for n,p in model.named_parameters() if "head.fc" not in n]
    optimizer = optim.AdamW([
        {"params":head_p,"lr":HEAD_LR},
        {"params":body_p,"lr":BODY_LR}
    ], weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS-WARMUP_EPOCH)
    scaler = GradScaler(); stopper=EarlyStop(20); crit=nn.CrossEntropyLoss()

    # == MixUp/CutMix (timm 0.9+ 关键字) ==
    from timm.data import Mixup
    mixup_fn = Mixup(
        num_classes     = NUM_CLASSES,
        mixup_alpha     = MIXUP_ALPHA,
        cutmix_alpha    = CUTMIX_ALPHA,
        label_smoothing = 0.1,
    )

    # == EMA & SWA ==
    ema = ModelEmaV2(model, decay=0.9998, device=DEVICE)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(optimizer, swa_lr=BODY_LR)

    get_ema_net = lambda ema_obj: ema_obj.module if hasattr(ema_obj,"module") else ema_obj

    # == helpers ==
    def train_epoch(ep):
        model.train(); TL,TA=AvgMeter(),AvgMeter()
        for st,(x,y) in enumerate(tqdm(train_loader,desc=f"[Train]{ep}/{EPOCHS}")):
            x,y = x.to(DEVICE),y.to(DEVICE)
            x,y = mixup_fn(x,y)        # batch 已偶数
            use_jr = (ep>=EPOCHS-20) and (st%JR_EVERY==0)
            if use_jr: x.requires_grad_(True)

            optimizer.zero_grad()
            with autocast():
                out=model(x); loss=crit(out,y)
                if use_jr:
                    g=torch.autograd.grad(out.sum(),x,create_graph=True)[0]
                    loss += JR_LAMBDA*g.pow(2).mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            ema.update(model)

            _,p=out.max(1)
            acc=(p.eq(y.argmax(1) if y.ndim==2 else y).float()).mean()
            TL.update(loss.item(),x.size(0)); TA.update(acc.item(),x.size(0))
        return TL.avg,TA.avg

    @torch.no_grad()
    def evaluate(loader, net):
        net.eval(); L,A=AvgMeter(),AvgMeter()
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            with autocast(): out=net(x); loss=crit(out,y)
            _,p=out.max(1); acc=p.eq(y).float().mean()
            L.update(loss.item(),x.size(0)); A.update(acc.item(),x.size(0))
        return L.avg,A.avg

    # == loop ==
    hist={k:[] for k in["tl","ta","vl","va"]}
    with open(LOGS/"log.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["ep","tl","ta","vl","va","lr"])
        best,bpath=0.0,None
        for ep in range(1,EPOCHS+1):
            if ep<=WARMUP_EPOCH:
                optimizer.param_groups[0]["lr"]=HEAD_LR*ep/WARMUP_EPOCH
            elif ep==WARMUP_EPOCH+1: scheduler.step(0)
            if ep==FREEZE_EPOCH+1:
                for p in body_p: p.requires_grad_(True)

            tl,ta = train_epoch(ep)
            vl,va = evaluate(val_loader, get_ema_net(ema))
            lr_now=optimizer.param_groups[0]["lr"]

            if ep>=SWA_START:
                swa_model.update_parameters(get_ema_net(ema)); swa_sched.step()
            else: scheduler.step()

            hist["tl"].append(tl); hist["ta"].append(ta)
            hist["vl"].append(vl); hist["va"].append(va)
            w.writerow([ep,tl,ta,vl,va,lr_now])
            print(f"Ep{ep}: TL{tl:.3f} TA{ta:.3f} | VL{vl:.3f} VA{va:.3f}")

            if va>best:
                best,bpath=va,CKPT/f"best_ep{ep:03d}_{va:.4f}.pth"
                torch.save({"state":get_ema_net(ema).state_dict(),"acc":va},bpath)
            if stopper(va): print("⏹ Early stop"); break

    update_bn(train_loader,swa_model)
    torch.save({"state":swa_model.state_dict(),"acc":best}, CKPT/"swa_best.pth")

    # == curves ==
    plt.figure(); plt.plot(hist["tl"],label="Train Loss"); plt.plot(hist["vl"],label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.grid(); plt.tight_layout()
    plt.savefig(FIGS/"loss_curve.png",dpi=150); plt.close()
    plt.figure(); plt.plot(hist["ta"],label="Train Acc"); plt.plot(hist["va"],label="Val Acc")
    plt.legend(); plt.xlabel("Epoch"); plt.grid(); plt.tight_layout()
    plt.savefig(FIGS/"acc_curve.png",dpi=150); plt.close()

    # == final test (SWA + flip TTA) ==
    @torch.no_grad()
    def tta_flip(x):
        f=torch.flip(x,dims=[3])
        out=swa_model(torch.cat([x,f],0).to(DEVICE)).softmax(1)
        return out[:x.size(0)]+out[x.size(0):]

    swa_model.eval(); TL,TA=AvgMeter(),AvgMeter()
    for x,y in tqdm(test_loader,desc="[Test]"):
        y=y.to(DEVICE); out=tta_flip(x); loss=crit(out,y)
        _,p=out.max(1); acc=p.eq(y).float().mean()
        TL.update(loss.item(),x.size(0)); TA.update(acc.item(),x.size(0))
    print(f"\n✅ Test ACC {TA.avg:.4f}  (best val {best:.4f})")
    print(f"🗂 Outputs → {OUT}")

# Windows-spawn guard
if __name__=="__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
