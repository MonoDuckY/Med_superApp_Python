import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from functools import partial
from skimage import transform, measure

# Import from the local segment_anything module
try:
    from segment_anything.modeling import (
        ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer,
    )
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

class MedSAM_InferenceModel:
    def __init__(self, weights_path: str = "weights/medsam_model_best.pth"):
        """
        Khởi tạo và load trọng số MedSAM.
        Chỉ nên khởi tạo 1 lần khi chạy Backend để tối ưu tốc độ.
        """
        self.weights_path = weights_path
        self.is_loaded = False
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        self.medsam_model = None
        self.load_model()

    def _build_medsam(self):
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        sam = Sam(
            image_encoder=ImageEncoderViT(
                depth=12, embed_dim=768, img_size=image_size, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12, patch_size=vit_patch_size, qkv_bias=True,
                use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
                window_size=14, out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3, iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
        return sam

    def load_model(self):
        if not HAS_SAM:
            print("⚠️ [CẢNH BÁO] Không tìm thấy package segment_anything. Chạy mock data.")
            return

        if not os.path.exists(self.weights_path):
            print(f"⚠️ [CẢNH BÁO] Không tìm thấy file trọng số MedSAM tại {self.weights_path}")
            print("⚠️ Module phân vùng sẽ trả về dữ liệu giả lập (mock data) cho đến khi file trọng số được thêm vào.")
            return

        print("Loading MedSAM model...")
        self.medsam_model = self._build_medsam().to(self.device)
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.medsam_model.load_state_dict(state_dict)
        self.medsam_model.eval()
        self.is_loaded = True
        print(f"✅ Đã load thành công model MedSAM trên {self.device}.")

    @torch.no_grad()
    def predict(self, image: np.ndarray, bbox: list[int] = None) -> dict:
        """
        Dự đoán vùng bệnh lý trên ảnh.
        bbox format: [xmin, ymin, xmax, ymax]
        """
        H, W = image.shape[:2]
        
        if not self.is_loaded:
            # Mock data nếu chưa có weights
            if bbox is None:
                bbox = [int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8)]
            return {
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "label": "myoma",
                        "bbox": bbox,
                        "segmentation": [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
                        "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                    }
                ]
            }

        if bbox is None:
            bbox = [0, 0, W, H]
            
        img_3c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for Image Encoder
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3,
            preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 1. Encode Image
        img_embed = self.medsam_model.image_encoder(tensor)
        
        # 2. Prepare prompt (box)
        box_np = np.array([bbox])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]
            
        sparse_emb, dense_emb = self.medsam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None,
        )
        
        # 3. Decode mask
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
        )
        sam_mask = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # 4. Extract polygon from mask
        contours = measure.find_contours(sam_mask, 0.5)
        polygon = []
        area = int(sam_mask.sum())
        
        if contours:
            contour = max(contours, key=len)
            polygon = [[float(c[1]), float(c[0])] for c in contour[::2]]
            
        return {
            "annotations": [
                {
                    "id": 1,
                    "category_id": 1,
                    "label": "medsam_seg",
                    "bbox": bbox,
                    "segmentation": polygon,
                    "area": area
                }
            ]
        }
