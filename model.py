# model.py
# 版本: 修改为接受 6D 特征输入 (XYZRGB)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.nn import knn_graph # 确保导入正确
except ImportError as e:
    print(f"FATAL Error importing PyTorch Geometric: {e}")
    import sys
    sys.exit(1)

class PyG_PointTransformerSegModel(nn.Module):
    """
    使用 PyG PointTransformerConv 的分割模型，修改为接受 6D 特征输入。
    """
    def __init__(self, num_classes, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.k = getattr(args, 'k_neighbors', 16)
        # !!! 修改点 1: 输入特征维度现在是 6 (XYZ+RGB) !!!
        input_feature_dim = 6
        embed_dim = getattr(args, 'embed_dim', 64)
        pt_hidden_dim = getattr(args, 'pt_hidden_dim', 128)
        pt_heads = getattr(args, 'pt_heads', 4)
        dropout_rate = getattr(args, 'dropout', 0.3)
        num_transformer_layers = getattr(args, 'num_transformer_layers', 2)

        # 1. 初始特征嵌入 MLP (输入维度改为 input_feature_dim)
        self.mlp_embed = nn.Sequential(
            # --- !!! 修改点 2: Linear 输入维度改为 6 !!! ---
            nn.Linear(input_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.bn_embed = nn.BatchNorm1d(embed_dim)

        # 2. PointTransformerConv 层 (结构不变，但接收的初始 x 特征维度是 embed_dim)
        self.transformer_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = embed_dim # Transformer 层接收的是嵌入后的特征
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                pyg_nn.PointTransformerConv(
                    in_channels=in_channels,
                    out_channels=pt_hidden_dim,
                    # heads=pt_heads, # 假设你更新了 PyG 或移除了这个参数
                    # dropout=dropout_rate
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(pt_hidden_dim))
            in_channels = pt_hidden_dim

        # 3. 逐点解码器 MLP (结构不变)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(pt_hidden_dim, pt_hidden_dim),
            nn.BatchNorm1d(pt_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pt_hidden_dim, num_classes)
        )

    # --- !!! 修改点 3: forward 方法输入和处理 !!! ---
    def forward(self, features):
        """
        模型的前向传播。

        Args:
            features (Tensor): 输入特征, 形状 (B, N, 6) - XYZRGB

        Returns:
            Tensor: 分割 logits, 形状 (B, N, num_classes)
        """
        B, N, C = features.shape
        if C != 6: # 现在期望 6 个通道
            raise ValueError(f"Input features should have 6 channels (XYZRGB), but got {C}")

        # --- 分离坐标 (pos) 和 准备初始特征 (x) ---
        pos = features[..., :3].contiguous() # 前 3 列是 XYZ，作为位置信息
        pos = pos.view(B * N, 3) # 转换为 (B*N, 3)
        features_flat = features.view(B * N, C) # (B*N, 6)
        batch_index = torch.arange(B, device=features.device).repeat_interleave(N) # (B*N,)

        # 1. 初始特征嵌入 (使用全部 6D 特征)
        x = self.mlp_embed(features_flat)
        x = F.relu(self.bn_embed(x)) # (B*N, embed_dim)

        # 2. 构建 k-NN 图 (使用 3D 坐标 pos)
        edge_index = knn_graph(pos, k=self.k, batch=batch_index, loop=False)

        # 3. 通过 PointTransformerConv 层
        #    输入 x 是嵌入特征，pos 是 3D 坐标
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x=x, pos=pos, edge_index=edge_index)
            x = F.relu(self.batch_norms[i](x))

        # 4. 逐点解码器
        logits_flat = self.decoder_mlp(x) # (B*N, num_classes)

        # 5. Reshape 输出
        logits = logits_flat.view(B, N, self.num_classes)

        return logits
    # --- 结束修改点 ---

# --- 测试代码块 (输入改为 6 维) ---
if __name__ == '__main__':
    class DummyArgs: # 需要包含所有模型 __init__ 用到的参数
        k_neighbors = 16; embed_dim = 64; pt_hidden_dim = 128
        pt_heads = 4; dropout = 0.3; num_transformer_layers = 2
        num_points = 1024 # 假设

    args = DummyArgs()
    num_classes_example = 50
    model = PyG_PointTransformerSegModel(num_classes=num_classes_example, args=args)
    print("PyG PointTransformer Segmentation Model (RGB input) instantiated.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dummy_input = torch.randn(4, args.num_points, 6) # (B, N, 6) <--- 输入维度改为 6
    print(f"Dummy input shape: {dummy_input.shape}")

    try: # 保留测试块的 try-except
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}") # 应为 (B, N, num_classes)
        assert output.shape == (4, args.num_points, num_classes_example)
        print("Forward pass test successful.")
    except Exception as e:
        print(f"Error during forward pass test: {e}"); import traceback; traceback.print_exc()