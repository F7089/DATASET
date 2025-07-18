# train.py  – herbal_medicine/
# ===============================================================
# ConvNeXt-Tiny (IN-22k) + RandAugment + MixUp + SWA + AMP
# 自动时间戳输出目录；Loss & Acc 曲线分开
# Windows-friendly (spawn)
# ===============================================================
import csv, random, datetime
from pathlib import Path
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment
from tqdm import tqdm
import matplotlib.pyplot as plt

# ───────── Hyper-params ─────────
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME   = "convnext_tiny.fb_in22k"
BATCH_SIZE   = 32        # 224×224
EPOCHS       = 120
WARMUP_EPOCH = 5
FREEZE_EPOCH = 10        # 前 10 epoch 只训练 head
SWA_START    = EPOCHS-30
MIXUP_ALPHA  = 0.4
LAMBDA_JR    = 0.0

HEAD_LR      = 3e-4
BODY_LR      = 3e-5
WEIGHT_DECAY = 1e-4

# ───────── Utils ─────────
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self,v,n=1): self.val=v; self.sum+=v*n; self.count+=n; self.avg=self.sum/self.count
class EarlyStopping:
    def __init__(self, patience=20): self.patience=patience; self.best=-1; self.counter=0
    def __call__(self, acc):
        if acc>self.best: self.best, self.counter = acc,0
        else: self.counter+=1
        return self.counter>=self.patience

# ───────── main() ─────────
def main():
    torch.manual_seed(SEED); random.seed(SEED)

    ROOT = Path(__file__).resolve().parent
    DATA = ROOT/"dataset"

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = ROOT/"outputs"/run_id
    CKPTDIR, LOGDIR, FIGDIR = [OUT/p for p in ("checkpoints","logs","figures")]
    for d in (CKPTDIR,LOGDIR,FIGDIR): d.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_tf = transforms.Compose([
        RandAugment(num_ops=2, magnitude=10),          # ★ 新增
        transforms.RandomResizedCrop(224, (0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_loader = DataLoader(datasets.ImageFolder(DATA/"train", train_tf),
                              BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(datasets.ImageFolder(DATA/"val",   eval_tf),
                              BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(datasets.ImageFolder(DATA/"test",  eval_tf),
                              BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    NUM_CLASSES = len(train_loader.dataset.classes)

    # ---- Model ----
    model = timm.create_model(MODEL_NAME, pretrained=True)
    in_feat = model.head.fc.in_features
    model.head.fc = nn.Linear(in_feat, NUM_CLASSES)
    model = model.to(DEVICE)

    # Freeze backbone initially
    for n,p in model.named_parameters():
        if "head.fc" not in n:
            p.requires_grad_(False)

    head_params = [p for n,p in model.named_parameters() if "head.fc" in n]
    body_params = [p for n,p in model.named_parameters() if "head.fc" not in n]

    optimizer = optim.AdamW([
        {"params": head_params, "lr":HEAD_LR},
        {"params": body_params,"lr":BODY_LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCH)
    scaler    = GradScaler()
    stopper   = EarlyStopping(patience=20)

    # SWA containers
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=BODY_LR)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # MixUp
    def mixup(x,y,a=MIXUP_ALPHA):
        if a<=0: return x,(y,y,1.)
        lam=random.betavariate(a,a); idx=torch.randperm(x.size(0),device=DEVICE)
        return lam*x+(1-lam)*x[idx], (y,y[idx],lam)
    def mix_loss(pred,t): y1,y2,lam=t; return lam*criterion(pred,y1)+(1-lam)*criterion(pred,y2)

    # History
    hist = {k:[] for k in ["tl","ta","vl","va"]}

    csv_path = LOGDIR/"log.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ep","tl","ta","vl","va","lr"])
        best_acc,best_path=0.0,None

        for ep in range(1,EPOCHS+1):
            # --- Warm-up LR ---
            if ep<=WARMUP_EPOCH:
                lr=HEAD_LR*ep/WARMUP_EPOCH
                optimizer.param_groups[0]["lr"]=lr
            elif ep==WARMUP_EPOCH+1:
                scheduler.step(0)

            # --- Unfreeze backbone ---
            if ep==FREEZE_EPOCH+1:
                for p in body_params: p.requires_grad_(True)

            # --- Train ---
            model.train(); TL,TA=AverageMeter(),AverageMeter()
            pbar=tqdm(train_loader,desc=f"[Train] {ep}/{EPOCHS}")
            for imgs,labels in pbar:
                imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                imgs,tgt=mixup(imgs,labels)

                optimizer.zero_grad()
                with autocast():
                    out=model(imgs); loss=mix_loss(out,tgt)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

                _,pred=out.max(1); lam=tgt[2]
                acc=((lam*pred.eq(tgt[0]).float())+((1-lam)*pred.eq(tgt[1]).float())).mean()
                TL.update(loss.item(),imgs.size(0)); TA.update(acc.item(),imgs.size(0))
                pbar.set_postfix(loss=f"{TL.avg:.3f}",acc=f"{TA.avg:.3f}")

            # --- Val ---
            model.eval(); VL,VA=AverageMeter(),AverageMeter()
            with torch.no_grad():
                for imgs,labels in tqdm(val_loader,desc=f"[Val ] {ep}/{EPOCHS}"):
                    imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
                    with autocast(): out=model(imgs); vloss=criterion(out,labels)
                    _,vp=out.max(1); vacc=vp.eq(labels).float().mean()
                    VL.update(vloss.item(),imgs.size(0)); VA.update(vacc.item(),imgs.size(0))

            lr_now=optimizer.param_groups[0]["lr"]
            print(f"Ep{ep}: TL{TL.avg:.3f} TA{TA.avg:.3f} | VL{VL.avg:.3f} VA{VA.avg:.3f} | lr {lr_now:.6f}")

            hist["tl"].append(TL.avg); hist["ta"].append(TA.avg)
            hist["vl"].append(VL.avg); hist["va"].append(VA.avg)
            w.writerow([ep,TL.avg,TA.avg,VL.avg,VA.avg,lr_now])

            # --- SWA update ---
            if ep >= SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # --- Checkpoint / Early stop ---
            if VA.avg>best_acc:
                best_acc=VA.avg
                best_path=CKPTDIR/f"best_ep{ep:03}_{best_acc:.4f}.pth"
                torch.save({"epoch":ep,"state":model.state_dict(),"best_acc":best_acc},best_path)
            if stopper(VA.avg): print("Early stopping."); break

    # ---- SWA BN update ----
    update_bn(train_loader, swa_model)
    torch.save({"state":swa_model.module.state_dict() if isinstance(swa_model,torch.nn.DataParallel) else swa_model.state_dict(),
                "best_acc":best_acc}, CKPTDIR/"swa_best.pth")

    # ---- Curves ----
    plt.figure(); plt.plot(hist["tl"],label="Train Loss"); plt.plot(hist["vl"],label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.grid(); plt.tight_layout()
    plt.savefig(FIGDIR/"loss_curve.png",dpi=150); plt.close()

    plt.figure(); plt.plot(hist["ta"],label="Train Acc"); plt.plot(hist["va"],label="Val Acc")
    plt.legend(); plt.xlabel("Epoch"); plt.grid(); plt.tight_layout()
    plt.savefig(FIGDIR/"acc_curve.png",dpi=150); plt.close()

    # ---- Test with SWA ----
    swa_model.eval(); TL,TA=AverageMeter(),AverageMeter()
    with torch.no_grad():
        for imgs,labels in tqdm(test_loader,desc="[Test]"):
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            with autocast(): out=swa_model(imgs); loss=criterion(out,labels)
            _,p=out.max(1); acc=p.eq(labels).float().mean()
            TL.update(loss.item(),imgs.size(0)); TA.update(acc.item(),imgs.size(0))
    print(f"\n✅ Test ACC {TA.avg:.4f}  (best val {best_acc:.4f})")
    print(f"🗂 Outputs → {OUT}")

# ---- Windows spawn guard ----
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
