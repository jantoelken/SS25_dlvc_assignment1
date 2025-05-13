## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code
import torch
from torch import nn

class Patcher(nn.Module):
    def __init__(self, patch_size):
        super(Patcher, self).__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        
    def forward(self, images):
        batch_size, channels, height, width = images.shape
        patch_height, patch_width = [self.patch_size, self.patch_size]
        assert height % patch_height == 0 and width % patch_width == 0, "Height and width must be divisible by the patch size."
        patches = self.unfold(images) #bs (cxpxp) N
        patches = patches.view(batch_size, channels, patch_height, patch_width, -1).permute(0, 4, 1, 2, 3) # bs N C P P
        return patches

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        # MultiheadAttention expects sequence first, then batch
        x_norm = x_norm.transpose(0, 1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = attn_out.transpose(0, 1)
        x = x + attn_out
        
        # Feedforward network
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class MyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, model_dim=100, num_heads=3, num_layers=2, num_classes=10):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # 1) Patching
        self.patcher = Patcher(patch_size=self.patch_size)
        
        # 2) Linear Projection
        self.linear_projector = nn.Linear(3 * self.patch_size ** 2, self.model_dim)
        
        # 3) Class Token
        self.class_token = nn.Parameter(torch.rand(1, 1, self.model_dim))
        
        # 4) Positional Embedding
        self.positional_embedding = nn.Parameter(torch.rand(1, (image_size // patch_size) ** 2 + 1, model_dim))
        
        # 5) Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.model_dim, self.num_heads) for _ in range(num_layers)
        ])
        
        # 6) Classification MLP
        self.mlp = nn.Linear(self.model_dim, self.num_classes)
    
    def forward(self, x):
        x = self.patcher(x)
        x = x.flatten(start_dim=2)
        x = self.linear_projector(x)
        
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        
        x = x + self.positional_embedding
        
        for block in self.blocks:
            x = block(x)
        
        # Using mean pooling as suggested in the article
        latent = x.mean(dim=1)
        logits = self.mlp(latent)
        
        return logits