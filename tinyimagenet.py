import os
import glob
from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class TinyImageNet(VisionDataset):
    """
    Tiny ImageNet-200 dataset with CIFAR-like interface.

    Key attributes to match CIFAR10/100:
        - self.data: np.ndarray (N, 64, 64, 3), dtype=uint8
        - self.targets: List[int] of length N
        - self.classes: List[str] (human-readable names)
        - self.class_to_idx: Dict[str, int] mapping class name -> integer label

    Args:
        root (str or Path): Root directory that *contains* the dataset folder.
            We will look for one of:
                - root / 'tinyimagenet'
                - root / 'tiny-imagenet-200'
                - root / 'tiny-imagenet'
        train (bool): If True -> 'train' split; else -> 'val' split.
        transform (callable, optional): transforms for images (PIL in -> tensor out)
        target_transform (callable, optional): transforms for integer targets
        download (bool): If True, download & extract Tiny ImageNet-200 into root.
        split (str, optional): One of {'train','val','test'}.
            Overrides `train` if provided. 'test' has no labels (targets=-1).
        base_folder (str, optional): Manually set if you keep a nonstandard folder name.

    Layout expected (standard Tiny ImageNet-200):
        tiny-imagenet-200/
          ├── wnids.txt
          ├── words.txt
          ├── train/
          │    └── <wnid>/
          │         ├── images/*.JPEG
          │         └── boxes.txt
          ├── val/
          │    ├── images/*.JPEG
          │    └── val_annotations.txt  (filename, wnid, x, y, w, h)
          └── test/
               └── images/*.JPEG
    """

    # Download info (for convenience; host sometimes throttles—use at your own risk)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    # No MD5 provided by the official site; leave as None to skip integrity on archive
    tgz_md5 = None

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split: Optional[str] = None,
        base_folder: Optional[str] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Resolve split to use (CIFAR uses train bool; we keep that + optional split)
        if split is None:
            self.split = "train" if train else "val"
        else:
            split = split.lower()
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be one of {'train','val','test'}")
            self.split = split
        self.train = (self.split == "train")  # keep CIFAR's attribute for repr

        # Determine dataset directory
        self.root = str(root)
        candidate_folders = []
        if base_folder:
            candidate_folders.append(base_folder)
        candidate_folders += ["tinyimagenet", "tiny-imagenet-200", "tiny-imagenet"]

        self.base_folder = None
        for cand in candidate_folders:
            p = os.path.join(self.root, cand)
            if os.path.isdir(p):
                self.base_folder = cand
                break

        if self.base_folder is None:
            # If download requested, do that first
            if download:
                self.download()
                # After download, the extracted folder is typically 'tiny-imagenet-200'
                extracted = os.path.join(self.root, "tiny-imagenet-200")
                if os.path.isdir(extracted):
                    self.base_folder = "tiny-imagenet-200"
                else:
                    # Try other fallbacks
                    for cand in ["tinyimagenet", "tiny-imagenet"]:
                        if os.path.isdir(os.path.join(self.root, cand)):
                            self.base_folder = cand
                            break

            if self.base_folder is None:
                raise RuntimeError(
                    "Tiny ImageNet dataset folder not found under root. "
                    "Expected one of: 'tinyimagenet', 'tiny-imagenet-200', 'tiny-imagenet'. "
                    "You can also pass base_folder=... if your folder is named differently."
                )

        # Load class metadata (wnids and words)
        dataset_dir = os.path.join(self.root, self.base_folder)
        wnids_path = os.path.join(dataset_dir, "wnids.txt")
        words_path = os.path.join(dataset_dir, "words.txt")

        if not os.path.isfile(wnids_path):
            raise RuntimeError(f"wnids.txt not found at {wnids_path}")
        if not os.path.isfile(words_path):
            raise RuntimeError(f"words.txt not found at {words_path}")

        self.wnids = self._read_wnids(wnids_path)  # list of 200 wnids
        wnid_to_words = self._read_words(words_path)  # wnid -> description string

        # Create classes (human-readable) aligned with wnids order
        self.classes: List[str] = [wnid_to_words.get(w, w) for w in self.wnids]
        # Map wnid -> label index (0..199)
        self.wnid_to_idx: Dict[str, int] = {w: i for i, w in enumerate(self.wnids)}
        # CIFAR-style class_to_idx uses class names; we keep it but tie to wnids order
        # to ensure uniqueness, prepend wnid when possible to avoid collisions.
        self.class_to_idx: Dict[str, int] = {
            f"{w} {wnid_to_words.get(w, w)}": i for i, w in enumerate(self.wnids)
        }

        # Load split
        if self.split == "train":
            data, targets = self._load_train_split(dataset_dir)
        elif self.split == "val":
            data, targets = self._load_val_split(dataset_dir)
        else:
            data, targets = self._load_test_split(dataset_dir)

        # CIFAR-compatible attributes
        self.data: Any = data  # np.ndarray (N, 64, 64, 3) uint8
        self.targets: List[int] = targets  # list of ints (or -1 for test)

    # ---------- Public API: same spirit as CIFAR ----------

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # ensure PIL Image like CIFAR

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train else ("Val" if self.split == "val" else "Test")
        return f"Split: {split}"

    # ---------- Integrity / Download ----------

    def _check_integrity(self) -> bool:
        """Lightweight presence checks for key files."""
        dataset_dir = os.path.join(self.root, self.base_folder)
        need = [
            os.path.join(dataset_dir, "wnids.txt"),
            os.path.join(dataset_dir, "words.txt"),
            os.path.join(dataset_dir, "train"),
            os.path.join(dataset_dir, "val"),
        ]
        for p in need:
            if not (os.path.exists(p)):
                return False
        return True

    def download(self) -> None:
        # If it looks already present, skip
        if any(os.path.isdir(os.path.join(self.root, cand)) for cand in
               ["tinyimagenet", "tiny-imagenet-200", "tiny-imagenet"]):
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    # ---------- Helpers to read metadata ----------

    @staticmethod
    def _read_wnids(path: str) -> List[str]:
        with open(path, "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
        return wnids

    @staticmethod
    def _read_words(path: str) -> Dict[str, str]:
        """
        words.txt lines look like: 'n01443537\ttench, Tinca tinca'
        """
        mapping = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()
        return mapping

    # ---------- Split loaders (build .data and .targets) ----------

    def _load_train_split(self, dataset_dir: str):
        """
        Iterate train/<wnid>/images/*.JPEG, map wnid -> integer label.

        Returns:
            data: np.ndarray (N, 64, 64, 3) uint8
            targets: List[int]
        """
        images: List[np.ndarray] = []
        targets: List[int] = []

        train_root = os.path.join(dataset_dir, "train")
        # ensure deterministic order: sort class dirs then filenames
        wnid_dirs = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
        for wnid in wnid_dirs:
            wnid_path = os.path.join(train_root, wnid, "images")
            # Some folders may have uppercase .JPEG; glob both
            img_paths = sorted(glob.glob(os.path.join(wnid_path, "*.JPEG")) + glob.glob(os.path.join(wnid_path, "*.jpg")) + glob.glob(os.path.join(wnid_path, "*.png")))
            label = self.wnid_to_idx.get(wnid)
            # Skip any unexpected wnid not in wnids.txt
            if label is None:
                continue
            for p in img_paths:
                arr = self._load_rgb64(p)
                images.append(arr)
                targets.append(label)

        if not images:
            raise RuntimeError(f"No training images found under {train_root}")
        data = np.stack(images, axis=0)  # (N, 64, 64, 3)
        return data, targets

    def _load_val_split(self, dataset_dir: str):
        """
        Validation: labels are in val/val_annotations.txt as:
            <filename> <wnid> x y w h
        Images live in val/images/.
        """
        val_root = os.path.join(dataset_dir, "val")
        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_root = os.path.join(val_root, "images")

        if not os.path.isfile(ann_path):
            raise RuntimeError(f"Validation annotations not found at {ann_path}")
        if not os.path.isdir(img_root):
            raise RuntimeError(f"Validation images folder not found at {img_root}")

        # Build filename -> wnid map
        fname_to_label: Dict[str, int] = {}
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if not parts or len(parts[0].split()) == 0:
                    # some copies have space-separated; handle both
                    parts = line.strip().split()
                if len(parts) >= 2:
                    fname, wnid = parts[0], parts[1]
                    if wnid in self.wnid_to_idx:
                        fname_to_label[fname] = self.wnid_to_idx[wnid]

        img_paths = sorted(glob.glob(os.path.join(img_root, "*.JPEG")) + glob.glob(os.path.join(img_root, "*.jpg")) + glob.glob(os.path.join(img_root, "*.png")))
        images: List[np.ndarray] = []
        targets: List[int] = []

        for p in img_paths:
            fname = os.path.basename(p)
            if fname not in fname_to_label:
                # ignore unknown files
                continue
            arr = self._load_rgb64(p)
            images.append(arr)
            targets.append(fname_to_label[fname])

        if not images:
            raise RuntimeError(f"No validation images found under {img_root}")
        data = np.stack(images, axis=0)
        return data, targets

    def _load_test_split(self, dataset_dir: str):
        """
        Test images are unlabeled in the official release.
        We return targets as -1 to keep shapes compatible.
        """
        test_root = os.path.join(dataset_dir, "test", "images")
        if not os.path.isdir(test_root):
            raise RuntimeError(f"Test images folder not found at {test_root}")

        img_paths = sorted(glob.glob(os.path.join(test_root, "*.JPEG")) + glob.glob(os.path.join(test_root, "*.jpg")) + glob.glob(os.path.join(test_root, "*.png")))
        if not img_paths:
            raise RuntimeError(f"No test images found under {test_root}")

        images: List[np.ndarray] = []
        for p in img_paths:
            images.append(self._load_rgb64(p))
        data = np.stack(images, axis=0)
        targets = [-1] * len(images)
        return data, targets

    @staticmethod
    def _load_rgb64(path: str) -> np.ndarray:
        """
        Load image, convert to RGB, ensure 64x64, return uint8 array (H, W, C).
        """
        with Image.open(path) as im:
            im = im.convert("RGB")
            # Most TinyImageNet images are already 64x64, but be safe.
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        return arr
