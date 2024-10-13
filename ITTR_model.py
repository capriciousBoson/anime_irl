# generator.py

import torch
import torch.nn as nn
from ITTR_pytorch import HPB, DPSA  # Assuming these are available from the library

class ITTRGenerator(nn.Module):
    def __init__(self, img_dim=512, num_blocks=9, heads=8, dim_head=32, top_k=16):
        super(ITTRGenerator, self).__init__()
        
        self.num_blocks = num_blocks
        # Add a convolutional layer to change input from 3 channels (RGB) to 512
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Create the series of HPB blocks
        self.hpb_blocks = nn.ModuleList([
            HPB(
                dim=img_dim,              # input dimension of image embeddings
                dim_head=dim_head,        # dimension per attention head
                heads=heads,              # number of attention heads
                attn_height_top_k=top_k,  # number of top indices to select for attention along height
                attn_width_top_k=top_k,   # number of top indices to select for attention along width
                attn_dropout=0.,          # attention dropout
                ff_mult=4,                # feedforward expansion factor
                ff_dropout=0.             # feedforward dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Final Dual Pruned Self-Attention (DPSA) layer
        self.dpsa = DPSA(
            dim=img_dim,          # input dimension of the image
            dim_head=dim_head,    # dimension per attention head
            heads=heads,          # number of attention heads
            height_top_k=top_k * 3,  # more top indices for final refinement
            width_top_k=top_k * 3,   # more top indices for final refinement
            dropout=0.            # dropout
        )
        
        # Final convolutional layer to return the image
        self.final_conv = nn.Conv2d(in_channels=img_dim, out_channels=3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Pass through the HPB blocks
        x = self.conv_in(x)
        for block in self.hpb_blocks:
            x = block(x)
        
        # Final pass through the DPSA block
        x = self.dpsa(x)
        
        # Final image generation
        x = self.final_conv(x)
        return self.tanh(x)

