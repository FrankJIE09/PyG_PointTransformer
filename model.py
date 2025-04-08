# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import knn_graph

class PyG_PointTransformerSegModel(nn.Module):
    """
    使用 PyG PointTransformerConv 构建的点云分割模型示例。
    """
    def __init__(self, num_classes, args=None):
        super().__init__()
        self.num_classes = num_classes
        # 从 args 获取超参数，提供默认值
        self.k = getattr(args, 'k_neighbors', 16)
        embed_dim = getattr(args, 'embed_dim', 64)
        pt_hidden_dim = getattr(args, 'pt_hidden_dim', 128)
        pt_heads = getattr(args, 'pt_heads', 4)
        dropout_rate = getattr(args, 'dropout', 0.3)
        num_transformer_layers = getattr(args, 'num_transformer_layers', 2) # 添加Transformer层数控制

        # 1. 初始特征嵌入 MLP
        self.mlp_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim) # 输出 embed_dim
        )
        self.bn_embed = nn.BatchNorm1d(embed_dim) # BatchNorm 在激活函数后

        # 2. PointTransformerConv 层 (使用 ModuleList)
        self.transformer_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                pyg_nn.PointTransformerConv(
                    in_channels=in_channels,
                    out_channels=pt_hidden_dim,
                    # heads=pt_heads,
                    # dropout=dropout_rate
                    # pos_nn 和 attn_nn 可以保持默认或根据需要定制
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(pt_hidden_dim))
            in_channels = pt_hidden_dim # 下一层的输入维度

        # 3. 逐点解码器 MLP
        self.decoder_mlp = nn.Sequential(
            nn.Linear(pt_hidden_dim, pt_hidden_dim), # 输入来自最后一个 transformer 层
            nn.BatchNorm1d(pt_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pt_hidden_dim, num_classes) # 输出 num_classes 个 logits
        )

    def forward(self, points):
        """
        Args:
            points (Tensor): 输入点云, 形状 (B, N, 3)
        Returns:
            Tensor: 分割 logits, 形状 (B, N, num_classes)
        """
        B, N, C = points.shape
        if C != 3:
            raise ValueError(f"Input points should have 3 channels (xyz), but got {C}")

        # 1. 构造 PyG 期望的输入格式
        pos = points.view(B * N, C) # (B*N, 3) 坐标作为位置信息
        batch_index = torch.arange(B, device=points.device).repeat_interleave(N) # (B*N,)

        # 2. 初始特征嵌入
        x = self.mlp_embed(pos)
        x = F.relu(self.bn_embed(x)) # (B*N, embed_dim)

        # 3. 动态构建 k-NN 图 (效率提示: 预处理或使用 Transform 更好)
        edge_index = knn_graph(pos, k=self.k, batch=batch_index, loop=False)

        # 4. 通过 PointTransformerConv 层
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x=x, pos=pos, edge_index=edge_index)
            x = F.relu(self.batch_norms[i](x)) # 应用 BatchNorm 和 ReLU

        # 5. 逐点解码器
        logits_flat = self.decoder_mlp(x) # (B*N, num_classes)

        # 6. Reshape 输出
        logits = logits_flat.view(B, N, self.num_classes)

        return logits

# --- 用于基本测试的示例代码块 ---
if __name__ == '__main__':
    class DummyArgs:
        k_neighbors = 16
        embed_dim = 64
        pt_hidden_dim = 128
        pt_heads = 4
        dropout = 0.3
        num_transformer_layers = 2
        # --- 模拟 train.py 中的参数 ---
        num_points = 1024 # 需要匹配数据加载

    args = DummyArgs()
    num_classes_example = 50 # ShapeNetPart
    model = PyG_PointTransformerSegModel(num_classes=num_classes_example, args=args)
    print("PyG PointTransformer Segmentation Model instantiated.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dummy_input = torch.randn(4, args.num_points, 3) # (B, N, C)
    print(f"Dummy input shape: {dummy_input.shape}")

    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}") # 应为 (B, N, num_classes)
        assert output.shape == (4, args.num_points, num_classes_example)
        print("Forward pass test successful.")
    except Exception as e:
        print(f"Error during forward pass test: {e}")
        import traceback
        traceback.print_exc()