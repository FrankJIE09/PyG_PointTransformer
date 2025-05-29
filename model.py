# model.py
# 版本: 修改为接受 6D 特征输入 (XYZRGB)
# 无需因 dataset.py 中的数据归一化而修改此文件。
# 但请确保 PointTransformerConv 参数设置正确。

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
        input_feature_dim = 6
        embed_dim = getattr(args, 'embed_dim', 64)
        pt_hidden_dim = getattr(args, 'pt_hidden_dim', 128)
        pt_heads = getattr(args, 'pt_heads', 4) # This is read from args
        dropout_rate = getattr(args, 'dropout', 0.3) # 确保这个参数被正确使用
        num_transformer_layers = getattr(args, 'num_transformer_layers', 2)

        self.mlp_embed = nn.Sequential(
            nn.Linear(input_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.bn_embed = nn.BatchNorm1d(embed_dim)

        self.transformer_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_transformer_layers):
            # !!! 关键回顾点: 确保 PyTorch Geometric 的 PointTransformerConv 参数正确 !!!
            # 根据您的 PyG 版本，heads 和 dropout 可能需要显式传递，或者有不同默认值。
            # 例如:
            self.transformer_layers.append(
                pyg_nn.PointTransformerConv(
                    in_channels=in_channels,
                    out_channels=pt_hidden_dim,
                    # heads=pt_heads, # <--- 确认此参数是否被正确使用或需要
                    # dropout=dropout_rate # <--- PointTransformerConv 可能不直接接受dropout，需查文档
                                          # dropout 更常用于 PointTransformerBlock 中的其他位置
                )
            )
            # 如果 PointTransformerConv 不直接接受 dropout，
            # 可以在其后添加 nn.Dropout(dropout_rate) 如果需要的话，
            # 但通常在 PointTransformer 块的特定位置使用。
            self.batch_norms.append(nn.BatchNorm1d(pt_hidden_dim))
            in_channels = pt_hidden_dim

        self.decoder_mlp = nn.Sequential(
            nn.Linear(pt_hidden_dim, pt_hidden_dim),
            nn.BatchNorm1d(pt_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Decoder 中的 Dropout
            nn.Linear(pt_hidden_dim, num_classes)
        )

    def forward(self, features):
        """
        模型的前向传播。
        Args:
            features (Tensor): 输入特征, 形状 (B, N, 6) - XYZRGB (经过归一化)
        Returns:
            Tensor: 分割 logits, 形状 (B, N, num_classes)
        """
        B, N, C = features.shape
        if C != 6:
            raise ValueError(f"Input features should have 6 channels (XYZRGB), but got {C}")

        pos = features[..., :3].contiguous() # 前 3 列是归一化后的 XYZ
        pos = pos.view(B * N, 3)
        features_flat = features.view(B * N, C) # (B*N, 6)
        batch_index = torch.arange(B, device=features.device).repeat_interleave(N)

        x = self.mlp_embed(features_flat)
        x = F.relu(self.bn_embed(x))

        edge_index = knn_graph(pos, k=self.k, batch=batch_index, loop=False)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x=x, pos=pos, edge_index=edge_index)
            x = F.relu(self.batch_norms[i](x))

        logits_flat = self.decoder_mlp(x)
        logits = logits_flat.view(B, N, self.num_classes)
        return logits

if __name__ == '__main__':
    class DummyArgs:
        k_neighbors = 16; embed_dim = 64; pt_hidden_dim = 128
        pt_heads = 4; dropout = 0.3; num_transformer_layers = 2
        num_points = 1024

    args = DummyArgs()
    num_classes_example = 50
    model = PyG_PointTransformerSegModel(num_classes=num_classes_example, args=args)
    print("PyG PointTransformer Segmentation Model (RGB input) instantiated.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 假设输入已经是归一化后的数据
    dummy_input = torch.randn(4, args.num_points, 6)
    print(f"Dummy input shape: {dummy_input.shape}")

    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (4, args.num_points, num_classes_example)
        print("Forward pass test successful.")
    except Exception as e:
        print(f"Error during forward pass test: {e}"); import traceback; traceback.print_exc()