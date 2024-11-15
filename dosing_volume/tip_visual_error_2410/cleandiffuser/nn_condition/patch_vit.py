import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViTImageCondition(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, transformer, channels=3):
        super().__init__()
        image_size_h, image_size_w = pair(image_size)
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = transformer

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding[:, :x.shape[1]]
        x = self.transformer(x)

        # Pooling the output tokens into a single vector of size (b, dim)
        x = x.mean(dim=1)
        return x
