# VGGT-Ω 完整推理流程 & 架构细节

> 本文档目标是：读完即可理解并大致复现整个项目。所有 shape 以
> `B=batch, N=帧数, D=embed_dim=1024, H=图像高, W=图像宽, P=patch_size=16` 标注。

---

## 一、总体架构

```
                        ┌──────────────────────────────────────────────┐
                        │              VGGTOmega (主模型)               │
                        │                                              │
  images [B,N,3,H,W]   │   ┌─────────────┐                            │
  ─────────────────────►   │ Aggregator  │  交替注意力编码器             │
        │                  │ (DINOv3 ViT │  输出: cached_layers         │
        │                  │  + 24层交替) │  + patch_token_start        │
        │                  └──────┬──────┘                             │
        │                         │                                    │
        │          ┌──────────────┼──────────────┐                     │
        │          ▼              ▼              ▼                     │
        │   ┌────────────┐ ┌───────────┐ ┌────────────────┐           │
        │   │ CameraHead │ │ DenseHead │ │TextAlignHead  │ (可选)      │
        │   └─────┬──────┘ └─────┬─────┘ └───────┬────────┘           │
        │         ▼              ▼               ▼                     │
        │   pose_enc [B,N,9] depth [B,N,H,W,1]  embedding [B,D]       │
        │                  conf  [B,N,H,W]                             │
        └──────────────────────────────────────────────┘
                        │
                        ▼
              后处理：pose解码 → 深度反投影 → GLB导出
```

---

## 二、① 图像预处理

**代码**: `vggt_omega/utils/load_fn.py → load_and_preprocess_images()`

### 输入
- `image_path_list`: 图像路径列表

### 处理步骤

| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 1. 加载 | PIL 读取 RGB；RGBA → 白色背景合成 RGB | PIL Image |
| 2. 宽高比裁剪 | 中心裁剪至 [0.5, 2.0] 范围 | PIL Image |
| 3. 缩放（两种模式） | | |
| &emsp;`balanced` | token 数 ≈ (resolution/P)²；按宽高比分配 h_patches, w_patches | |
| &emsp;`max_size` | 最长边 = resolution，短边按比例缩放并对齐到 P 的整数倍 | |
| 4. 对齐填充 | 若各图尺寸不一，pad 到最大尺寸（constant=1.0） | |
| 5. ToTensor | PIL → float tensor, 值域 [0, 1] | |

### 输出
```
images: Tensor [N, 3, H, W]   # float32, 值域 [0, 1]
# 注: 此时还没有 batch 维度，VGGTOmega.forward() 会 unsqueeze(0) 变成 [1, N, 3, H, W]
```

### balanced 模式的计算
```python
token_number = (image_resolution // patch_size) ** 2   # 例: (512/16)² = 1024
w_patches = round(sqrt(token_number / aspect_ratio))   # aspect_ratio = H/W
h_patches = round(token_number / w_patches)
target_h = h_patches * patch_size
target_w = w_patches * patch_size
```

---

## 三、② Aggregator — 聚合编码器

**代码**: `vggt_omega/models/aggregator.py`

这是 VGGT-Ω 最核心的模块。由两部分组成：
1. **DINOv3 ViT Backbone**（24 层 ViT-Large） — 提取每张图的 patch 特征
2. **交替注意力网络**（24 层 frame_block + inter_frame_block） — 跨帧交互

### 3.1 DINOv3 Vision Transformer Backbone

**代码**: `vggt_omega/models/layers/vision_transformer.py → DinoVisionTransformer`

> ⚠️ 关键细节：虽然变量名叫 `patch_embed`，但它不是一个简单的 Conv2d，而是一个**完整的 24 层 ViT-Large**，包含自己的 cls_token、storage_tokens、24 层 SelfAttentionBlock 和 LayerNorm。

#### 构造参数
```python
DinoVisionTransformer(
    img_size=224, patch_size=16, in_chans=3,
    embed_dim=1024, depth=24, num_heads=16,    # ViT-Large
    ffn_ratio=4,                                # FFN hidden = 4096
    qkv_bias=True, proj_bias=True, ffn_bias=True,
    drop_path_rate=0.0,
    layerscale_init=1e-5,
    norm_layer="layernormbf16",                 # LayerNorm(eps=1e-5)
    ffn_layer="mlp",                            # 标准 MLP (GELU)
    n_storage_tokens=4,                         # DINO 的 register tokens
    mask_k_bias=True,                           # K 的 bias 被 NaN mask
    pos_embed_rope_base=100.0,
    pos_embed_rope_normalize_coords="max",
    pos_embed_rope_dtype="fp32",
)
```

#### 内部结构
```
DinoVisionTransformer
├── PatchEmbed                    # Conv2d(3, 1024, kernel=16, stride=16)
├── cls_token                     # [1, 1, 1024], init N(0, 0.02)
├── storage_tokens                # [1, 4, 1024], init N(0, 0.02)
├── mask_token                    # [1, 1024], init zeros
├── rope_embed                    # RopePositionEmbedding
├── blocks[0..23]                 # 24 × SelfAttentionBlock
└── norm                          # LayerNorm(1024, eps=1e-5)
```

#### 前向传播（被 Aggregator 调用时）

```python
# 输入: images_flat [B*N, 3, H, W]

# 1. Patch Embedding (Conv2d)
x = Conv2d(images_flat)                    # [B*N, 1024, H/16, W/16]
x = x.flatten(2).transpose(1,2)            # [B*N, H/16*W/16, 1024]
# 例: 512×512 图像 → [B*N, 1024, 1024]

# 2. 拼接 cls + storage + patch tokens
x = cat([cls_token, storage_tokens, x], dim=1)
#   [B*N, 1+4+1024, 1024] = [B*N, 1029, 1024]

# 3. 24 层 SelfAttentionBlock (带 RoPE)
for blk in blocks:
    x = blk(x, rope=(sin, cos))
#   x 保持 [B*N, 1029, 1024]

# 4. LayerNorm
x = norm(x)                                # [B*N, 1029, 1024]

# 5. 分离并返回 dict
x_norm_patchtokens = x[:, 5:]              # [B*N, 1024, 1024]  ← 只用这个！
# cls_token 和 storage_tokens 的输出被丢弃
```

**关键**：Aggregator 只取 `x_norm_patchtokens`，DINOv3 的 cls_token 和 storage_tokens 不参与后续计算。DINOv3 backbone 相当于一个强大的"特征提取器"。

#### 权重初始化
```python
# Linear: trunc_normal_(std=0.02), bias=zeros
# LayerNorm: reset_parameters()
# LayerScale: gamma = 1e-5 (常数)
# cls_token: N(0, 0.02)
# storage_tokens: N(0, 0.02)
# mask_token: zeros
# PatchEmbed Conv2d: uniform(-sqrt(k), sqrt(k)), k = 1/(3*16*16)
# mask_k_bias 的 bias_mask: Q部分=1, K部分=NaN(被mask), V部分=1
```

---

### 3.2 Aggregator 的交替注意力网络

#### 可学习参数

| 参数 | Shape | 初始化 | 说明 |
|------|-------|--------|------|
| `camera_token` | `[1, 2, 1, 1024]` | N(0, 1e-3) | dim1 的 2 个元素分别用于首帧和其余帧 |
| `register_token` | `[1, 2, 16, 1024]` | N(0, 1e-3) | 同上，16 个 register tokens |
| `frame_blocks[0..23]` | 24 × SelfAttentionBlock | 见下文 | 帧内自注意力 |
| `inter_frame_blocks[0..23]` | 24 × SelfAttentionBlock | 见下文 | 帧间注意力 |

#### Token 构成（每帧）

```
每帧 token 序列 (共 num_tokens 个):
┌──────────────┬─────────────────────────┬─────────────────────────────────┐
│ camera_token │   register_tokens × 16  │    patch_tokens (来自DINOv3)     │
│    1 个      │        16 个            │      H/16 × W/16 个             │
└──────────────┴─────────────────────────┴─────────────────────────────────┘
│◄──── patch_token_start = 17 ────►│
│◄───────────── num_tokens = 17 + (H×W)/256 ─────────────────────────────►│

# 例: 512×512 图像 → num_tokens = 17 + 1024 = 1041
```

#### 首帧 vs 其余帧的参数分离

```python
# camera_token shape: [1, 2, 1, D]
#                    ─  ─  ─  ─
#                    │  │  │  └── embed_dim
#                    │  │  └───── 1 个 camera token
#                    │  └──────── 2 组参数 (索引0=首帧, 索引1=其余帧)
#                    └─────────── batch=1

def slice_expand_and_flatten(token_tensor, B, N):
    first  = token_tensor[:, 0:1]     # [1, 1, 1, D] → 首帧参数
    others = token_tensor[:, 1:]      # [1, 1, 1, D] → 其余帧参数
    first  = first.expand(B, 1, ...)  # [B, 1, 1, D]
    others = others.expand(B, N-1, ...) # [B, N-1, 1, D]
    tokens = cat([first, others], dim=1) # [B, N, 1, D]
    return tokens.reshape(B*N, 1, D)    # [B*N, 1, D]
```

#### 前向传播完整 shape 追踪

```python
# 输入: images [B, N, 3, H, W]

# ── Step 0: 图像归一化 ──
images = (images - resnet_mean) / resnet_std
#   resnet_mean = [0.485, 0.456, 0.406]  (broadcast [1,1,3,1,1])
#   resnet_std  = [0.229, 0.224, 0.225]
images_flat = images.reshape(B*N, 3, H, W)
#   [B*N, 3, H, W]

# ── Step 1: 展开 camera & register tokens ──
camera_tokens   = slice_expand_and_flatten(self.camera_token, B, N)
#   [B*N, 1, D]
register_tokens = slice_expand_and_flatten(self.register_token, B, N)
#   [B*N, 16, D]

# ── Step 2: DINOv3 Backbone 提取 patch features ──
patch_tokens = self.patch_embed(images_flat)  # 完整 ViT-L 前向
#   返回 dict → 取 "x_norm_patchtokens"
patch_tokens = patch_tokens["x_norm_patchtokens"]
#   [B*N, num_patches, D]    num_patches = (H/16) × (W/16)

# ── Step 3: 拼接所有 tokens ──
tokens = cat([camera_tokens, register_tokens, patch_tokens], dim=1)
#   [B*N, 1+16+num_patches, D] = [B*N, num_tokens, D]

# ── Step 4: 计算 RoPE ──
patch_grid_size = (H//16, W//16)
rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
#   各 [num_patches, D_head]  D_head = D/num_heads = 64
#   RoPE 以 fp32 计算，不随模型 dtype 变化

# ── Step 5: 24 层交替注意力 ──
outputs = []
for block_idx in range(24):
    # ── 5a: Frame Block (帧内) ──
    tokens_2d = tokens.reshape(B*N, num_tokens, D)
    tokens_2d = frame_blocks[block_idx](tokens_2d, rope=(sin, cos))
    #   SelfAttentionBlock 内部:
    #     x = x + LayerScale(Attention(LayerNorm(x), rope))
    #     x = x + LayerScale(FFN(LayerNorm(x)))
    #   Attention: QKV linear → split heads → QK norm → RoPE → SDPA → proj
    #   RoPE 只对后 num_patches 个 token 生效 (prefix=17 不加 RoPE)
    frame_tokens = tokens_2d.reshape(B, N, num_tokens, D)

    # ── 5b: Inter-Frame Block (帧间) ──
    tokens_4d = tokens.reshape(B, N, num_tokens, D)

    if block_idx in [2, 6, 9, 14, 20]:
        # "register" 模式:
        cam_reg = tokens_4d[:, :, :17].reshape(B, N*17, D)
        patches = tokens_4d[:, :, 17:].reshape(B, N*(num_tokens-17), D)
        cam_reg = inter_frame_blocks[block_idx](cam_reg, rope=None)
        #   只有 camera+register tokens 跨帧交互
        #   patch tokens 不参与！
        tokens = cat([cam_reg, patches], dim=1)
        #   重组回 [B, N, num_tokens, D]
    else:
        # "global" 模式:
        tokens_2d = tokens_4d.reshape(B, N*num_tokens, D)
        tokens_2d = inter_frame_blocks[block_idx](tokens_2d, rope=None)
        #   所有帧的所有 token 全局交互
        #   rope=None: 帧间注意力不用位置编码

    # ── 5c: 缓存中间层 ──
    if block_idx in {4, 11, 17, 23}:
        # 拼接 frame_tokens 和 inter_frame_tokens (在 embed 维度)
        cached = cat([frame_tokens, tokens_4d], dim=-1)
        #   [B, N, num_tokens, 2*D] = [B, N, num_tokens, 2048]
        outputs.append(cached)
    else:
        outputs.append(None)

# 输出:
#   outputs: list[24] — 4 个非 None tensor [B, N, num_tokens, 2048]
#            分别在 layer 4, 11, 17, 23
#   patch_token_start: int = 17
```

---

### 3.3 SelfAttentionBlock 详解

**代码**: `vggt_omega/models/layers/block.py`

这是整个模型中重复最多的基本单元。

#### 结构

```
输入 x [B, T, D]
  │
  ├──► LayerNorm(x) ──► SelfAttention(·, rope) ──► LayerScale(·) ──► + x  →  x_attn
  │                                                                    │
  └──► LayerNorm(x_attn) ──► FFN(·) ──► LayerScale(·) ──► + x_attn  →  x_out
```

公式：
```
x_attn = x + LayerScale₁(Attention(LayerNorm₁(x), rope))
x_out  = x_attn + LayerScale₂(FFN(LayerNorm₂(x_attn)))
```

#### SelfAttention 内部

**代码**: `vggt_omega/models/layers/attention.py`

```python
# 输入 x [B, T, D],  rope = (sin, cos) 或 None

# 1. QKV Projection
qkv = Linear(x)                       # [B, T, 3D]
#   mask_k_bias=True 时使用 LinearKMaskedBias:
#     K 部分的 bias 被 NaN mask 覆盖 → 等效于 K 无 bias
#     bias_mask: [Q部分=1, K部分=NaN, V部分=1]

# 2. Split heads
q, k, v = qkv.reshape(B, T, 3, 16, 64).unbind(2).transpose(1,2)
#   q, k, v 各 [B, 16, T, 64]    # 16 heads, head_dim=64

# 3. QK Norm (仅 Aggregator 的 frame/inter blocks 使用)
q = LayerNorm(q)    # per-head LayerNorm(64)
k = LayerNorm(k)

# 4. RoPE (仅 frame blocks 使用，inter blocks rope=None)
#   RoPE 只对后 num_patches 个 token 生效:
prefix = T - num_patches              # = 17 (camera + register)
q_prefix = q[:, :, :prefix, :]        # 不加 RoPE
q_patch  = rope_apply(q[:, :, prefix:, :], sin, cos)
q = cat([q_prefix, q_patch])
#   k 同理

# 5. Scaled Dot-Product Attention
out = F.scaled_dot_product_attention(q, k, v)
#   [B, 16, T, 64]

# 6. Output projection
out = out.transpose(1,2).reshape(B, T, D)
out = Linear(out)                     # [B, T, D]
```

#### RoPE 旋转位置编码

**代码**: `vggt_omega/models/layers/rope_position_encoding.py`

```python
# RopePositionEmbedding 参数:
#   embed_dim=1024, num_heads=16, base=100
#   normalize_coords="max", dtype=fp32

# D_head = 1024 / 16 = 64
# periods = base^(2*i / (D_head/2)),  i ∈ [0, D_head/4)
#         = 100^(2*i / 16),  共 16 个周期值

# 坐标生成 (normalize_coords="max"):
max_HW = max(H_patches, W_patches)
coords_h = (arange(0.5, H_patches) / max_HW)    # 归一化到 [0, 1]
coords_w = (arange(0.5, W_patches) / max_HW)
# → meshgrid → flatten → [H_patches*W_patches, 2]
# → 映射到 [-1, +1]

# 角度计算:
angles = 2π * coords[:, :, None] / periods[None, None, :]
#   [num_patches, 2, D_head/4]
angles = angles.flatten(1, 2)          # [num_patches, D_head/2]
angles = angles.tile(2)               # [num_patches, D_head]  ← 复制一遍

sin = sin(angles)                      # [num_patches, D_head]
cos = cos(angles)                      # [num_patches, D_head]

# RoPE 应用 (在 Attention 中):
# x_rotated = x * cos + rotate_half(x) * sin
# rotate_half: [x0,x1,...,x31,x32,...,x63] → [-x32,...,-x63,x0,...,x31]
```

**轴向（Axial）设计**：H 和 W 坐标共享同一套 periods，但通过 `tile(2)` 复制，前 D/2 维编码 H 方向，后 D/2 维编码 W 方向。

#### FFN (Mlp)

**代码**: `vggt_omega/models/layers/ffn_layers.py`

```python
class Mlp:
    fc1 = Linear(D, D*4)        # 1024 → 4096
    act = GELU()
    drop = Dropout(0)
    fc2 = Linear(D*4, D)        # 4096 → 1024
    # forward: fc1 → GELU → drop → fc2 → drop
```

#### LayerScale

**代码**: `vggt_omega/models/layers/layer_scale.py`

```python
class LayerScale:
    gamma = Parameter(ones(D) * 1e-5)    # 逐通道缩放，初始值很小
    # forward: x * gamma
```

---

## 四、③ Camera Head — 相机位姿预测

**代码**: `vggt_omega/models/heads/camera_head.py`

### 输入
```python
aggregated_tokens_list[-1]   # 最后一层的缓存输出
#   shape: [B, N, num_tokens, 2048]  (frame_tokens ‖ inter_frame_tokens 拼接)
patch_token_start = 17
```

### 处理流程 & Shape

```python
# 1. 提取 camera + register tokens
tokens = aggregated_tokens_list[-1][:, :, :17, :]
#   [B, N, 17, 2048]

# 2. LayerNorm + 转 fp32
tokens = tokens.float()
tokens = LayerNorm(tokens)              # LayerNorm(2048, eps=1e-5)
#   [B, N, 17, 2048]

# 3. 跨帧展平
tokens = tokens.reshape(B, N*17, 2048)
#   [B, N*17, 2048]

# 4. 4 层 SelfAttentionBlock (无 RoPE, 无 QK norm)
#   每层: x = x + Attention(LN(x)) + FFN(LN(x))
#   配置: dim=2048, heads=16, ffn_ratio=4, mask_k_bias=True
for block in trunk:                     # 4 个 block
    tokens = block(tokens, rope=None)
#   [B, N*17, 2048]

# 5. 重组 + 取 camera token
tokens = tokens.reshape(B, N, 17, 2048)
camera_tokens = LayerNorm(tokens[:, :, 0, :])
#   [B, N, 2048]   ← 只有第一个 (camera) token

# 6. MLP 回归
out = Linear(2048, 1024) → GELU → Linear(1024, 9)
#   [B, N, 9]

# 7. 激活函数
translation = raw[..., :3]              # 无激活
quaternion  = raw[..., 3:7]             # 无激活 (后续标准化)
fov         = ReLU(raw[..., 7:]) + 0.01 # 保证正值
#   [B, N, 9]  = [B, N, (3T + 4Q + 2FoV)]
```

### 输出
```python
pose_enc: [B, N, 9]   # float32
```

---

## 五、④ Dense Head — 深度 & 置信度预测

**代码**: `vggt_omega/models/heads/dense_head.py`

> 灵感来自 Depth-Anything-V2 的多尺度特征融合架构。

### 输入
```python
aggregated_tokens_list   # 4 个非 None 缓存 (layer 4, 11, 17, 23)
images                   # [B, N, 3, H, W]
patch_token_start = 17
```

### 5.1 多尺度特征提取

从 4 个缓存层提取 patch tokens，投影到不同通道数并缩放到统一空间分辨率：

```python
# 配置
intermediate_layer_idx = [4, 11, 17, 23]
out_channels = [256, 512, 1024, 1024]
# resize_scale = [4×, 2×, 1×, 0.5×]

# 假设输入 512×512 → patch_grid = 32×32

for feature_idx, layer_idx in enumerate([4, 11, 17, 23]):
    x = aggregated_tokens_list[layer_idx]
    #   [B, N, num_tokens, 2048]

    x = x[:, :, 17:]                   # 只取 patch tokens
    #   [B, N, num_patches, 2048]

    x = x.float()
    x = x.reshape(B*N, num_patches, 2048)
    x = LayerNorm(x)                   # LayerNorm(2048, eps=1e-5)
    x = x.permute(0,2,1).reshape(B*N, 2048, 32, 32)
    #   [B*N, 2048, 32, 32]

    # 1×1 卷积投影到目标通道
    x = Conv2d(2048, out_ch, 1×1)(x)
    #   feature 0: [B*N, 256, 32, 32]
    #   feature 1: [B*N, 512, 32, 32]
    #   feature 2: [B*N, 1024, 32, 32]
    #   feature 3: [B*N, 1024, 32, 32]

    # 添加正弦位置编码 (不是 RoPE!)
    x = x + sinusoidal_pos_embed(x.shape[-2:], x.shape[1])
    #   sin/cos 位置编码, ratio=0.1, omega_0=100

    # 缩放到统一分辨率
    x = resize_layer(x)
    #   feature 0: ConvTranspose2d(k=4,s=4) → [B*N, 256, 128, 128]   4× 上采样
    #   feature 1: ConvTranspose2d(k=2,s=2) → [B*N, 512, 64, 64]     2× 上采样
    #   feature 2: Identity                  → [B*N, 1024, 32, 32]   不变
    #   feature 3: Conv2d(k=3,s=2,p=1)       → [B*N, 1024, 16, 16]   2× 下采样
```

### 5.2 Scratch Network — 由粗到细融合

```python
# 先各自通过 3×3 Conv 统一到 256 通道
layer_1_rn = Conv2d(256, 256, 3, 1, 1)(feature_0)   # [B*N, 256, 128, 128]
layer_2_rn = Conv2d(512, 256, 3, 1, 1)(feature_1)   # [B*N, 256, 64, 64]
layer_3_rn = Conv2d(1024, 256, 3, 1, 1)(feature_2)  # [B*N, 256, 32, 32]
layer_4_rn = Conv2d(1024, 256, 3, 1, 1)(feature_3)  # [B*N, 256, 16, 16]

# 由粗到细的渐进式融合 (每个 refinenet = 2× ResidualConvUnit + 上采样)
#
# ResidualConvUnit:
#   ReLU → Conv3×3 → ReLU → Conv3×3 → + residual

out = refinenet4(layer_4_rn)
#   [B*N, 256, 16, 16] → 上采样到 (32,32)
out = refinenet3(out, layer_3_rn)
#   out + ResConvUnit(layer_3_rn) → ResConvUnit → 上采样到 (64,64)
out = refinenet2(out, layer_2_rn)
#   out + ResConvUnit(layer_2_rn) → ResConvUnit → 上采样到 (128,128)
out = refinenet1(out, layer_1_rn)
#   out + ResConvUnit(layer_1_rn) → ResConvUnit
#   [B*N, 256, 128, 128]
```

### 5.3 预测头

```python
# 添加位置编码
fused = fused + sinusoidal_pos_embed(...)

# 深度预测
depth_logits = Conv2d(256, (P/4)², 1×1)(fused)
#   P/4 = 4, (P/4)² = 16
#   [B*N, 16, 128, 128]
depth_logits = pixel_shuffle(depth_logits, 4)
#   [B*N, 1, 512, 512]  ← 恢复到原图分辨率！
depth_logits = depth_logits.permute(0,2,3,1)
#   [B*N, 512, 512, 1]

depth = exp(depth_logits)
#   [B*N, H, W, 1]  正值深度

# 置信度预测 (独立 Conv 头)
conf_logits = Conv2d(256, 16, 1×1)(fused)
#   权重初始化: zeros, bias = log(1.05 - 1.0) ≈ -2.94
#   → 初始时 conf ≈ 1.05
conf_logits = pixel_shuffle(conf_logits, 4)
conf_logits = conf_logits.permute(0,2,3,1).squeeze(-1)
#   [B*N, 512, 512]

depth_conf = 1 + exp(conf_logits)
#   [B*N, H, W]  始终 ≥ 1.0

# 重组
depth = depth.reshape(B, N, H, W, 1)      # float32
depth_conf = depth_conf.reshape(B, N, H, W) # float32
```

### 5.4 分块推理

```python
# DenseHead.forward() 支持 frames_chunk_size 参数 (默认 8)
# 将 N 帧按 chunks 分批处理以节省显存:
for start in range(0, N, 8):
    end = min(start+8, N)
    depth_chunk, conf_chunk = _forward_impl(images[:, start:end])
depth = cat(depth_chunks, dim=1)
depth_conf = cat(conf_chunks, dim=1)
```

---

## 六、⑤ 后处理

### 6.1 相机位姿解码

**代码**: `vggt_omega/utils/pose_enc.py → encoding_to_camera()`

```python
# 输入: pose_enc [B, N, 9]

T = pose_enc[..., :3]              # 平移 [B, N, 3]
quat = pose_enc[..., 3:7]          # 四元数 XYZW (scalar-last) [B, N, 4]
fov_h = pose_enc[..., 7]           # 垂直 FoV [B, N]
fov_w = pose_enc[..., 8]           # 水平 FoV [B, N]

# 四元数 → 旋转矩阵 (XYZW 顺序)
R = quat_to_mat(quat)              # [B, N, 3, 3]
#   详见 rotation.py: 使用 PyTorch3D 风格的四元数到矩阵转换
#   输出标准化: 确保实部 (W) 非负

# 外参 (camera-from-world, OpenCV 坐标系)
extrinsics = cat([R, T.unsqueeze(-1)], dim=-1)
#   [B, N, 3, 4]

# 内参
fy = (H/2) / tan(fov_h / 2)
fx = (W/2) / tan(fov_w / 2)
intrinsics = zeros(B, N, 3, 3)
intrinsics[..., 0, 0] = fx         # focal_x
intrinsics[..., 1, 1] = fy         # focal_y
intrinsics[..., 0, 2] = W/2        # principal_point_x
intrinsics[..., 1, 2] = H/2        # principal_point_y
intrinsics[..., 2, 2] = 1.0
```

### 6.2 深度反投影到 3D 世界坐标

**代码**: `demo_gradio.py → unproject_depth_map_to_point_map()`

```python
# 输入:
#   depth_map [N, H, W]     (depth[..., 0])
#   extrinsic [N, 3, 4]     (camera-from-world)
#   intrinsic [N, 3, 3]

# 1. 生成像素坐标网格
y, x = meshgrid(H, W)                 # 各 [H, W]
x = broadcast_to([N, H, W])
y = broadcast_to([N, H, W])

# 2. 像素 → 相机坐标系 3D 点
X_cam = (x - cx) / fx × depth        # [N, H, W]
Y_cam = (y - cy) / fy × depth        # [N, H, W]
Z_cam = depth                         # [N, H, W]
camera_points = stack([X, Y, Z], dim=-1)
#   [N, H, W, 3]

# 3. 相机坐标 → 世界坐标
#   world = R^T × (camera_point - T)
R = extrinsic[:, :3, :3]              # [N, 3, 3]
T = extrinsic[:, :3, 3]              # [N, 3]
world_points = einsum("sij,shwj->shwi", R^T, camera_points - T)
#   [N, H, W, 3]
```

### 6.3 GLB 场景导出

**代码**: `visual_util.py → predictions_to_glb()`

#### 点云过滤

```python
# 1. 深度边缘过滤
#    检测相邻像素深度相对跳变 > 3% 的位置
edge_mask = depth_edge(depth, rtol=0.03, kernel_size=3)
conf[edge_mask] = 0.0

# 2. 置信度百分位数过滤
threshold = percentile(conf, conf_thres)   # 默认 conf_thres=20%
mask = conf >= threshold

# 3. 有限值过滤
mask &= isfinite(vertices).all(axis=1)
mask &= conf > 1e-5

# 4. 可选背景过滤
if mask_black_bg:  mask &= colors.sum(axis=1) >= 16
if mask_white_bg:  mask &= ~(R>240 & G>240 & B>240)

# 5. 可选天空过滤 (ONNX skyseg 模型)
if mask_sky:       conf *= sky_mask > 0.1

# 6. 均匀采样至 max_points 个点
indices = linspace(0, len(vertices)-1, max_points)
```

#### 场景坐标变换

```python
# 以第一帧相机为原点，应用 OpenGL 坐标系转换
opengl_matrix = diag(1, -1, -1, 1)     # Y/Z 轴翻转
scene.apply_transform(inv(extrinsics[0]) @ opengl_matrix)
```

#### 相机可视化

```python
# 每个相机位姿用一个 4 面锥体标记
cone_width = scene_scale * 0.05
cone_height = scene_scale * 0.1
# 锥体经过以下变换:
#   1. 绕 Z 轴旋转 45° (使锥体尖端朝前)
#   2. 沿 Z 轴下移 cone_height
#   3. OpenGL 坐标变换
#   4. 相机 world_from_camera 变换
# 锥体使用 3 层顶点（原始 + 0.95× + 轻微旋转2°）产生厚度
# 颜色: gist_rainbow colormap, 按帧索引均匀分布
```

---

## 七、AMP 混合精度策略

**代码**: `vggt_omega/models/vggt_omega.py → forward()`

```python
# Aggregator: 使用 bf16 (或 fp16) 自动混合精度
amp_dtype = bf16 if cuda_supports_bf16 else fp16
with autocast(device_type="cuda", dtype=amp_dtype):
    aggregated_tokens_list, patch_token_start = aggregator(images)

# Heads: 强制 fp32
with autocast(device_type="cuda", enabled=False):   # 关闭 autocast
    pose_enc = camera_head(...)      # fp32
    depth, conf = dense_head(...)    # fp32
```

---

## 八、完整数据流 Shape 汇总

以 **B=1, N=10 帧, 512×512 输入图像, patch_size=16** 为例：

```
输入: images [1, 10, 3, 512, 512]
        │
        ▼ 归一化 (ResNet mean/std)
  images_flat [10, 3, 512, 512]
        │
        ▼ DINOv3 ViT-L (24层)
  patch_tokens [10, 1024, 1024]      # 1024 = 32×32 patches
        │
        ▼ 拼接 camera(1) + register(16) + patch(1024)
  tokens [10, 1041, 1024]
        │
        ▼ 24 层交替注意力
  frame_tokens   [1, 10, 1041, 1024]  # 每层输出
  inter_tokens   [1, 10, 1041, 1024]  # 每层输出
        │
        ▼ 缓存层 cat (dim=-1)
  cached[layer]  [1, 10, 1041, 2048]  # 层 4, 11, 17, 23
        │
   ┌────┴────┐
   ▼         ▼
CameraHead   DenseHead
   │         │
   │ 提取前17 tokens
   │ [1,10,17,2048]
   │ → norm → reshape
   │ [1, 170, 2048]
   │ → 4层 attention
   │ → 取 camera token
   │ [1, 10, 2048]
   │ → MLP
   ▼         │
pose_enc     │ 提取 patch tokens (4层)
[1,10,9]     │ → 投影+缩放+融合
   │         │ → pixel_shuffle
   ▼         ▼
解码为:    depth      [1, 10, 512, 512, 1]
extrinsics depth_conf [1, 10, 512, 512]
[1,10,3,4]     │
intrinsics     ▼
[1,10,3,3]   反投影 → world_points [10, 512, 512, 3]
   │             │
   └──────┬──────┘
          ▼
    predictions_to_glb() → scene.glb
```

---

## 九、权重初始化总结

| 模块 | 参数 | 初始化方式 |
|------|------|-----------|
| DINOv3 backbone (Linear) | weight | `trunc_normal_(std=0.02)` |
| DINOv3 backbone (Linear) | bias | `zeros` |
| DINOv3 cls_token | `[1,1,1024]` | `N(0, 0.02)` |
| DINOv3 storage_tokens | `[1,4,1024]` | `N(0, 0.02)` |
| DINOv3 mask_token | `[1,1024]` | `zeros` |
| DINOv3 PatchEmbed Conv2d | weight | `uniform(-√k, √k)`, `k=1/(3×256)` |
| LayerScale (所有模块) | gamma | `constant(1e-5)` |
| Aggregator camera_token | `[1,2,1,1024]` | `N(0, 1e-3)` |
| Aggregator register_token | `[1,2,16,1024]` | `N(0, 1e-3)` |
| Aggregator frame/inter blocks | 同 SelfAttentionBlock | 见下 |
| SelfAttentionBlock (Linear) | weight | `trunc_normal_(std=0.02)` |
| SelfAttentionBlock (Linear) | bias | `zeros` (K bias 被 mask) |
| SelfAttentionBlock (LayerNorm) | weight/bias | `reset_parameters()` |
| CameraHead camera_branch | MLP | 默认 PyTorch init |
| CameraHead token_norm | LayerNorm | `reset_parameters()` |
| DenseHead proj_conf | Conv2d weight | `zeros` |
| DenseHead proj_conf | Conv2d bias | `constant(log(0.05))` ≈ -2.996 |
| DenseHead 其他 Conv2d | weight | 默认 PyTorch init |
| TextAlignmentHead language_token | `[1,1,2048]` | `trunc_normal_(std=0.02)` |

---

## 十、TextAlignment Head（可选）

**代码**: `vggt_omega/models/heads/text_alignment_head.py`

> 仅 `VGGTOmega(enable_alignment=True)` + 256px checkpoint 使用。

```python
# 输入: aggregated_tokens_list[-1][:, :, :17, :]   # [B, N, 17, 2048]

# 1. LayerNorm
tokens = LayerNorm(camera_and_register_tokens)
#   [B, N, 17, 2048]

# 2. 添加可学习的 language_token
tokens = tokens.reshape(B, N*17, 2048)
language_token = self.language_token.expand(B, 1, 2048)
readout = cat([language_token, tokens], dim=1)
#   [B, 1 + N*17, 2048]

# 3. 4 层 SelfAttentionBlock (无 RoPE)
for block in readout_blocks:
    readout = block(readout, rope=None)

# 4. 取 language token + 投影
lang_emb = LayerNorm(readout[:, 0])
#   [B, 2048]
embedding = Linear(2048, 1024) → GELU → LN → Linear(1024, 2048)
#   [B, 2048]
embedding = F.normalize(embedding, dim=-1)
#   L2 归一化的文本对齐嵌入
```

---

## 十一、坐标系说明

| 坐标系 | 约定 | 使用场景 |
|--------|------|---------|
| **OpenCV** | X→右, Y→下, Z→前 | 相机外参 (camera-from-world) |
| **OpenGL** | X→右, Y→上, Z→后 | GLB 场景文件 |
| **四元数** | XYZW (scalar-last) | 模型内部 9D 编码 |

转换：`opengl_matrix = diag(1, -1, -1, 1)` 将 OpenCV 坐标转为 OpenGL 坐标。

---

## 十二、项目文件 → 模块对应

```
vggt-omega/
├── demo_gradio.py                  # Gradio Demo (上传→推理→GLB可视化)
├── visual_util.py                  # GLB 导出 (predictions_to_glb)
│
├── vggt_omega/
│   ├── models/
│   │   ├── vggt_omega.py           # VGGTOmega: 组合 Aggregator + Heads
│   │   ├── aggregator.py           # Aggregator: DINOv3 backbone + 交替注意力
│   │   ├── heads/
│   │   │   ├── camera_head.py      # CameraHead: 9D 位姿回归
│   │   │   ├── dense_head.py       # DenseHead: 多尺度深度预测
│   │   │   ├── text_alignment_head.py  # TextAlignmentHead: 文本嵌入
│   │   │   └── utils.py            # sinusoidal pos embed, UV grid
│   │   └── layers/
│   │       ├── attention.py        # SelfAttention (RoPE, QK norm, mask_k_bias)
│   │       ├── block.py            # SelfAttentionBlock (Pre-norm + LS + residual)
│   │       ├── ffn_layers.py       # Mlp (GELU), SwiGLUFFN
│   │       ├── layer_scale.py      # LayerScale (gamma=1e-5)
│   │       ├── patch_embed.py      # PatchEmbed (Conv2d)
│   │       ├── rms_norm.py         # RMSNorm
│   │       ├── rope_position_encoding.py  # 轴向 RoPE
│   │       ├── vision_transformer.py      # DinoVisionTransformer (完整 ViT)
│   │       └── utils.py            # cat_keep_shapes, named_apply 等
│   └── utils/
│       ├── load_fn.py              # 图像加载与预处理
│       ├── pose_enc.py             # 9D ↔ extrinsics/intrinsics 编解码
│       ├── geometry.py             # SE(3) 逆矩阵
│       └── rotation.py             # 四元数 ↔ 旋转矩阵 (XYZW)
│
├── examples/                       # 4 个示例视频
├── requirements.txt                # torch, torchvision, numpy, PIL, einops, ...
├── requirements_demo.txt           # gradio, scipy, trimesh, matplotlib, ...
└── pyproject.toml                  # 包定义
```
