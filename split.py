"""
Split 'train/' into new 'train/' + 'test/':

Assumed current layout
herbal_medicine/
├── dataes/
│   ├── train/     # any number of imgs/class
│   └── val/       # 19 imgs/class (keep)
└── ...

Result layout
herbal_medicine/
└── split_dataset/
    ├── train/<class_x>/   # original train minus 20
    ├── val/<class_x>/     # identical copy of original val
    └── test/<class_x>/    # 20 sampled imgs / class
"""
import random, shutil
from pathlib import Path

# ---------- CONFIG -------------------------------------------------
ROOT          = Path("herbal_medicine/dataes")    # 原始数据根
SRC_TRAIN_DIR = ROOT / "train"
SRC_VAL_DIR   = ROOT / "val"
DST_ROOT      = ROOT.parent / "split_dataset"     # 输出根
TEST_NUM      = 20                                # 每类抽出的数量
SEED          = 42
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}
# -------------------------------------------------------------------

random.seed(SEED)

def copy_imgs(img_paths, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in img_paths:
        shutil.copy2(p, dst_dir / p.name)

# 1️⃣ 处理 train → train + test
for cls_dir in SRC_TRAIN_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    if len(imgs) < TEST_NUM:
        print(f"⚠️  Skip '{cls_dir.name}': only {len(imgs)} images (< {TEST_NUM})")
        continue

    test_imgs  = random.sample(imgs, TEST_NUM)
    train_imgs = [p for p in imgs if p not in test_imgs]

    copy_imgs(train_imgs, DST_ROOT / "train" / cls_dir.name)
    copy_imgs(test_imgs,  DST_ROOT / "test"  / cls_dir.name)

# 2️⃣ 复制 val 原样过去
for cls_dir in SRC_VAL_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    copy_imgs(imgs, DST_ROOT / "val" / cls_dir.name)

print("\n✅ Split finished.")
print(f"Train & test saved under: {DST_ROOT}")
print("  train/  -> 原 train 减去 20 张")
print("  test/   -> 每类 20 张")
print("  val/    -> 原样复制")
