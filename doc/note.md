# VGGT-Ω 笔记

## 项目简介

VGGT-Ω（Visual Geometry Group Transformer - Omega）是由牛津大学 VGG 和 Meta AI 联合开发的前馈式 3D 重建模型。它能够从一组图像或视频中，**一次前向传播**即可预测所有相机的位姿和逐像素深度，无需传统的迭代优化（如 SfM/MVS）。

- **论文**: [arXiv:2605.15195](https://arxiv.org/abs/2605.15195)
- **项目主页**: [vggt-omega.github.io](http://vggt-omega.github.io/)
- **在线 Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/facebook/vggt-omega)
- **GitHub**: [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)

## 核心特点

1. **前馈推理** — 不需要迭代优化，单次前向传播即可得到相机位姿和深度
2. **交替注意力** — 帧内自注意力 + 帧间全局注意力交替进行，有效建模多视图关系
3. **DINOv2 Backbone** — 利用预训练视觉大模型的强特征表示能力
4. **可扩展性** — 支持 1~500+ 帧输入，显存近似线性增长
5. **文本对齐（可选）** — 256px checkpoint 支持文本-视觉对齐嵌入

## 快速使用

```bash
# 安装
pip install -r requirements.txt
pip install -e .

# 运行 Demo
pip install -r requirements_demo.txt
python demo_gradio.py --checkpoint path/to/model.pt --image-resolution 512
```

## 最小推理代码

```python
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load("path/to/model.pt", map_location="cpu"))

images = load_and_preprocess_images(["img1.png", "img2.png"], image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
depth = predictions["depth"]           # [B, N, H, W, 1]
depth_conf = predictions["depth_conf"] # [B, N, H, W]
```

## 详细流程文档

请参阅 [pipeline.md](./pipeline.md) 了解完整的推理流程和模型架构。
