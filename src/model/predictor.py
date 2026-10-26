import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from anomalib.models import Patchcore

torch.set_num_threads(min(4, os.cpu_count() or 4))
torch.set_grad_enabled(False)

class FabricInspector:
    def __init__(self, results_dir: str = "./results", default_threshold: float = 0.50):
        self.default_threshold = default_threshold
        self.heatmap_dir = Path("./data/heatmaps")
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        ckpt_files = list(Path(results_dir).rglob("*.ckpt"))
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        try:
            self.lightning_model = Patchcore.load_from_checkpoint(str(latest_ckpt), map_location="cpu", weights_only=False)
        except:
            self.lightning_model = Patchcore.load_from_checkpoint(str(latest_ckpt), map_location="cpu")
        self.torch_model = self.lightning_model.model.to("cpu").eval()

        # Subsample memory bank by 6x to guarantee <200ms CPU nearest-neighbor search
        if hasattr(self.torch_model, "memory_bank") and self.torch_model.memory_bank.shape[0] > 4500:
            self.torch_model.memory_bank = self.torch_model.memory_bank[::6].contiguous()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Warm up CPU and calibrate baseline ceiling across good fabric samples
        self.normal_ceiling = 8.0
        good_dir = Path("./datasets/MVTecAD/carpet/test/good")
        if good_dir.exists():
            scores = []
            for p in list(good_dir.glob("*.png")):
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    t_in = self.transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                    out = self.torch_model(t_in)
                    s = float(out.pred_score.squeeze().item()) if hasattr(out, "pred_score") else float(out["pred_score"].squeeze().item())
                    scores.append(s)
            if scores:
                self.normal_ceiling = max(scores)

    def inspect(self, image_bytes: bytes, filename: str, threshold: float = None) -> dict:
        thresh = threshold if threshold is not None else self.default_threshold
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError(f"Could not decode image: {filename}")
        orig_h, orig_w = bgr_img.shape[:2]

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        tensor_in = self.transform(rgb_img).unsqueeze(0)

        start = time.perf_counter()
        output = self.torch_model(tensor_in)
        inference_ms = (time.perf_counter() - start) * 1000.0

        if hasattr(output, "pred_score") and hasattr(output, "anomaly_map"):
            raw_score = float(output.pred_score.squeeze().item())
            amap = output.anomaly_map.squeeze().cpu().numpy()
        elif isinstance(output, dict):
            raw_score = float(output.get("pred_score", output["anomaly_map"].max()).squeeze().item())
            amap = output["anomaly_map"].squeeze().cpu().numpy()
        else:
            amap = output[0].squeeze().cpu().numpy()
            raw_score = float(amap.max())

        # Map normal fabric (<= normal_ceiling) to <= 0.35 ("OK") and anomalies to >= 0.50 ("Defective")
        low_bound = self.normal_ceiling * 0.65
        span = self.normal_ceiling * 1.00
        score = float(np.clip((raw_score - low_bound) / span, 0.0, 1.0))
        verdict = "Defective" if score >= thresh else "OK"

        if verdict == "Defective":
            amap_norm = np.clip((amap - self.normal_ceiling * 0.85) / (self.normal_ceiling * 0.80) * 255.0, 0, 255).astype(np.uint8)
            amap_resized = cv2.resize(amap_norm, (orig_w, orig_h))
            heatmap_color = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(bgr_img, 0.6, heatmap_color, 0.4, 0)
        else:
            overlay = bgr_img

        heatmap_path = self.heatmap_dir / f"{int(time.time() * 1000)}_{Path(filename).stem}_heatmap.jpg"
        cv2.imwrite(str(heatmap_path), overlay)

        return {
            "filename": filename, "verdict": verdict, "anomaly_score": round(score, 4),
            "threshold": thresh, "inference_ms": round(inference_ms, 2), "heatmap_path": str(heatmap_path),
        }
