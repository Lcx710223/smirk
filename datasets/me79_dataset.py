import os
import json
import random
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from skimage.transform import estimate_transform, warp
import albumentations as A
from skimage import transform as trans

# -----------------------
# Utility functions
# -----------------------
def create_mask(landmarks, shape):
    landmarks = landmarks.astype(np.int32)[..., :2]
    hull = cv2.convexHull(landmarks)
    mask = np.ones(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 0)
    return mask

mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53,
                     55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380,
                     381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133,
                     144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                     168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358,
                     0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
                     87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
                     308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
                     415]

# -----------------------
# BasePrep class
# -----------------------
class BasePrep:
    def __init__(self, config, test=False):
        # config expected to have: train.image_size, train.train_scale_min, train.train_scale_max, train.test_scale
        self.config = config
        self.image_size = getattr(config.train, "image_size", getattr(config, "image_size", 224))
        self.test = test
        if not self.test:
            self.scale = [config.train.train_scale_min, config.train.train_scale_max]
        else:
            self.scale = config.train.test_scale

        self.transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
                A.CLAHE(p=0.255),
                A.RGBShift(p=0.25),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.9),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               additional_targets={'mediapipe_keypoints': 'keypoints'})

        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

    def estimate_norm(self, lmk, image_size=112, mode='arcface'):
        assert lmk.shape == (5, 2)
        assert image_size % 112 == 0 or image_size % 128 == 0
        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio
        dst = self.arcface_dst * ratio
        dst[:, 0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M

    @staticmethod
    def crop_face(frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

    def prepare_data(self, image, landmarks_fan, landmarks_mediapipe):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if landmarks_fan is None:
            flag_landmarks_fan = False
            landmarks_fan = np.zeros((68, 2))
        else:
            flag_landmarks_fan = True

        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale

        tform = self.crop_face(image, landmarks_mediapipe, scale, image_size=self.image_size)
        landmarks_mediapipe = landmarks_mediapipe[..., :2]

        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
        cropped_landmarks_fan = np.dot(tform.params, np.hstack([landmarks_fan, np.ones([landmarks_fan.shape[0], 1])]).T).T
        cropped_landmarks_fan = cropped_landmarks_fan[:, :2]

        cropped_landmarks_mediapipe = np.dot(tform.params, np.hstack([landmarks_mediapipe, np.ones([landmarks_mediapipe.shape[0], 1])]).T).T
        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[:, :2]

        hull_mask = create_mask(cropped_landmarks_mediapipe, (self.image_size, self.image_size))

        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[mediapipe_indices, :2]

        if not self.test:
            transformed = self.transform(
                image=cropped_image,
                mask=1 - hull_mask,
                keypoints=cropped_landmarks_fan,
                mediapipe_keypoints=cropped_landmarks_mediapipe
            )

            cropped_image = (transformed['image'] / 255.0).astype(np.float32)
            cropped_landmarks_fan = np.array(transformed['keypoints']).astype(np.float32)
            cropped_landmarks_mediapipe = np.array(transformed['mediapipe_keypoints']).astype(np.float32)
            hull_mask = 1 - transformed['mask']
        else:
            cropped_image = (cropped_image / 255.0).astype(np.float32)
            cropped_landmarks_fan = cropped_landmarks_fan.astype(np.float32)
            cropped_landmarks_mediapipe = cropped_landmarks_mediapipe.astype(np.float32)

        cropped_landmarks_fan[:, :2] = cropped_landmarks_fan[:, :2] / self.image_size * 2 - 1
        cropped_landmarks_mediapipe[:, :2] = cropped_landmarks_mediapipe[:, :2] / self.image_size * 2 - 1
        masked_cropped_image = cropped_image * hull_mask[..., None]

        cropped_image = cropped_image.transpose(2, 0, 1)
        masked_cropped_image = masked_cropped_image.transpose(2, 0, 1)
        hull_mask = hull_mask[..., None]
        hull_mask = hull_mask.transpose(2, 0, 1)

        # mica image
        landmarks_arcface_crop = landmarks_fan[[36, 45, 32, 48, 54]].copy()
        landmarks_arcface_crop[0] = (landmarks_fan[36] + landmarks_fan[39]) / 2
        landmarks_arcface_crop[1] = (landmarks_fan[42] + landmarks_fan[45]) / 2

        tform = self.estimate_norm(landmarks_arcface_crop, 112)

        image_norm = image / 255.0
        mica_image = cv2.warpAffine(image_norm, tform, (112, 112), borderValue=0.0)
        mica_image = mica_image.transpose(2, 0, 1)

        image_t = torch.from_numpy(cropped_image).type(dtype=torch.float32)
        masked_image = torch.from_numpy(masked_cropped_image).type(dtype=torch.float32)
        landmarks_fan_t = torch.from_numpy(cropped_landmarks_fan).type(dtype=torch.float32)
        landmarks_mediapipe_t = torch.from_numpy(cropped_landmarks_mediapipe).type(dtype=torch.float32)
        hull_mask_t = torch.from_numpy(hull_mask).type(dtype=torch.float32)
        mica_image_t = torch.from_numpy(mica_image).type(dtype=torch.float32)

        data_dict = {
            'img': image_t,
            'landmarks_fan': landmarks_fan_t[..., :2],
            'flag_landmarks_fan': flag_landmarks_fan,
            'landmarks_mp': landmarks_mediapipe_t[..., :2],
            'mask': hull_mask_t,
            'img_mica': mica_image_t
        }
        return data_dict

# -----------------------
# Utility loaders
# -----------------------
def _load_json_optional(path: Optional[str]):
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _load_fan_txt_optional(path: Optional[str]):
    """
    Load 68-fan landmarks from TXT68FAN .txt
    Accepts comma-separated or whitespace-separated 136 numbers (x1 y1 x2 y2 ...).
    Returns ndarray shape (68,2) or None.
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # split by comma or whitespace
        if ',' in content:
            parts = [p.strip() for p in content.replace('\n', ',').split(',') if p.strip() != '']
        else:
            parts = content.split()
        vals = np.array([float(x) for x in parts], dtype=np.float32)
        if vals.size == 136:
            return vals.reshape(68, 2)
        # Some files might be JSON-like with list of pairs; try eval-safe parse
        try:
            js = json.loads(content)
            arr = np.asarray(js, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 68:
                return arr[:, :2]
        except Exception:
            pass
        return None
    except Exception:
        return None

# -----------------------
# Me79Dataset
# -----------------------
class Me79Dataset(Dataset, BasePrep):
    """
    Dataset expecting root with:
      IMAGES/         -> <id>.<ext>
      TXT68FAN/       -> <id>.txt
      MP468JSON/      -> <id>.json
    """
    def __init__(self, root: str, config, list_file: Optional[str] = None, items: Optional[List[str]] = None, test: bool = False):
        self.root = root
        self.image_dir = os.path.join(root, "IMAGES")
        self.fan_dir = os.path.join(root, "TXT68FAN")
        self.mp_dir = os.path.join(root, "MP468JSON")

        if items is not None:
            ids = items
        elif list_file is not None:
            assert os.path.exists(list_file), f"list_file not found: {list_file}"
            with open(list_file, 'r', encoding='utf-8') as f:
                ids = [l.strip() for l in f if l.strip()]
        else:
            if not os.path.isdir(self.image_dir):
                raise ValueError("images directory not found and no list_file/items provided")
            imgs = [os.path.splitext(fn)[0] for fn in os.listdir(self.image_dir)
                    if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
            ids = sorted(imgs)

        self.data_list = []
        for pid in ids:
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(self.image_dir, pid + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break

            # adapt to your naming
            fan_path = os.path.join(self.fan_dir, pid + ".txt")
            mp_path  = os.path.join(self.mp_dir,  pid + ".json")

            self.data_list.append({
                "id": pid,
                "img_path": img_path if img_path is not None else None,
                "fan_path": fan_path if os.path.exists(fan_path) else None,
                "mp_path": mp_path if os.path.exists(mp_path) else None
            })

        BasePrep.__init__(self, config, test=test)

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index: int) -> Optional[Dict[str, Any]]:
        entry = self.data_list[index]
        pid = entry["id"]

        # image path fallback
        img_path = entry["img_path"]
        if img_path is None:
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(self.image_dir, pid + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id {pid}")

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"cv2 failed to read image for id {pid} at {img_path}")

        # fan/mp path fallback
        fan_path = entry.get("fan_path")
        if fan_path is None:
            cand = os.path.join(self.fan_dir, pid + ".txt")
            fan_path = cand if os.path.exists(cand) else None

        mp_path = entry.get("mp_path")
        if mp_path is None:
            cand = os.path.join(self.mp_dir, pid + ".json")
            mp_path = cand if os.path.exists(cand) else None

        # load landmarks
        fan_lm = _load_fan_txt_optional(fan_path)
        mp_json = _load_json_optional(mp_path)

        landmarks_fan = None
        landmarks_mp = None

        if fan_lm is not None:
            try:
                lf = np.asarray(fan_lm, dtype=np.float32)
                if lf.ndim == 1 and lf.size == 136:
                    lf = lf.reshape(68, 2)
                landmarks_fan = lf
            except Exception:
                landmarks_fan = None

        if mp_json is not None:
            try:
                if isinstance(mp_json, dict) and "landmarks" in mp_json:
                    lm = np.asarray(mp_json["landmarks"], dtype=np.float32)
                else:
                    lm = np.asarray(mp_json, dtype=np.float32)
                if lm.ndim == 1:
                    lm = lm.reshape(-1, 2)
                landmarks_mp = lm
            except Exception:
                landmarks_mp = None

        # synthesize mp from fan if missing
        if landmarks_mp is None:
            mp_synth = np.zeros((468, 3), dtype=np.float32)
            if landmarks_fan is not None:
                try:
                    mp_synth[:68, :2] = landmarks_fan
                    landmarks_mp = mp_synth
                except Exception:
                    landmarks_mp = None
            else:
                landmarks_mp = None

        # final guard: without mediapipe landmarks we can't crop; drop sample
        if landmarks_mp is None:
            return None

        data_dict = self.prepare_data(image, landmarks_fan, landmarks_mp)
        data_dict["meta_id"] = pid
        return data_dict

    def __getitem__(self, index):
        max_tries = 10
        tries = 0
        while tries < max_tries:
            try:
                data = self.__getitem_aux__(index)
                if data is not None:
                    lf = data.get("landmarks_fan", None)
                    valid = False
                    if lf is not None:
                        if isinstance(lf, torch.Tensor):
                            valid = (lf.shape[0] == 68)
                        elif isinstance(lf, np.ndarray):
                            valid = (lf.shape[0] == 68)
                        else:
                            valid = hasattr(lf, "shape") and lf.shape[0] == 68
                    if valid:
                        return data
                    else:
                        print(f"[DEBUG] Invalid landmark shape for ID={self.data_list[index]['id']}")
                else:
                    print(f"[DEBUG] __getitem_aux__ returned None for ID={self.data_list[index]['id']}")

                # 随机换一个 index 重试
                index = random.randint(0, len(self.data_list) - 1)
                tries += 1
                print("Warning: re-sampling due to invalid sample, new index", index)

            except Exception as e:
                pid = self.data_list[index]['id'] if index < len(self.data_list) else 'OUT_OF_RANGE'
                print(f"Error in loading sample ID={pid} (index={index}), retrying... Exception: {e}")
                index = random.randint(0, len(self.data_list) - 1)
                tries += 1

        # 如果多次尝试仍失败，返回完整占位，避免 collate 报错
        print(f"[ERROR] Failed to load a valid sample after {max_tries} retries for index {index}")
        return {
            "img": torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32),
            "landmarks_fan": torch.zeros(68, 2, dtype=torch.float32),
            "landmarks_mp": torch.zeros(len(mediapipe_indices), 2, dtype=torch.float32),
            "mask": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "img_mica": torch.zeros(3, 112, 112, dtype=torch.float32),
            "meta_id": None,
            "flag_landmarks_fan": False
        }
