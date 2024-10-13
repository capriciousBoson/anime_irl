# generator.py

import torch
import torch.nn as nn
from ITTR_pytorch import HPB, DPSA  # Assuming these are available from the library

class ITTRGenerator(nn.Module):
    def __init__(self, img_dim=512, num_blocks=9, heads=8, dim_head=32, top_k=16):
        super(ITTRGenerator, self).__init__()
        
        self.num_blocks = num_blocks
        # Add a convolutional layer to change input from 3 channels (RGB) to 512
       
        self.conv_layers = nn.Sequential(
            # 7x7 Convolution, Instance Norm, GELU
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # Output: (batch_size, 64, 128, 128)
            nn.InstanceNorm2d(64),
            nn.GELU(),

            # 3x3 Convolution, stride=2, Instance Norm, GELU
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  # Output: (batch_size, 256, 64, 64)
            nn.InstanceNorm2d(256),
            nn.GELU(),

            # 3x3 Convolution, stride=2, Instance Norm, GELU
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # Output: (batch_size, 512, 32, 32)
            )
        
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
        # self.dpsa = DPSA(
        #     dim=img_dim,          # input dimension of the image
        #     dim_head=dim_head,    # dimension per attention head
        #     heads=heads,          # number of attention heads
        #     height_top_k=top_k * 3,  # more top indices for final refinement
        #     width_top_k=top_k * 3,   # more top indices for final refinement
        #     dropout=0.            # dropout
        # )
        
        # Final convolutional layer to return the image
        self.final_conv = nn.Conv2d(in_channels=img_dim, out_channels=3, kernel_size=3, padding=1)

        # DECODER
        self.deocder_layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Output channels: 128
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
        
        # Layer 2: Upsample
        self.deocder_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 32 -> 64
        
        # Layer 3: 3x3 Conv -> IN -> GELU
        self.deocder_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output channels: 128
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
        
        # Layer 4: Upsample
        self.deocder_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64 -> 128
        
        # Layer 5: 3x3 Conv -> IN -> GELU
        self.deocder_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Output channels: 256
            nn.InstanceNorm2d(128),
            nn.GELU()
        )
        
        # Layer 6: Upsample
        self.deocder_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128 -> 256
        
        # Layer 7: 7x7 Conv
        self.deocder_layer4 = nn.Conv2d(128, 3, kernel_size=7, padding=3)  # Output channels: 3

        # Final activation layer: Tanh
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Pass through the HPB blocks
        print(f"shape of input at forward : {x.shape} -------------")
        x = self.conv_layers(x)
        print(f"shape of output after conv2d layers : {x.shape} -------------")
        for block in self.hpb_blocks:
            x = block(x)
        
        print(f"shape of output after hpb blocks : {x.shape} -----------")
        # Final pass through the DPSA block
        # x = self.dpsa(x)

        # print(f"shape of output after dpsa blocks : {x.shape} -----------")
        # Final image generation
        x = self.deocder_layer1(x)                 # Shape: (B, 128, 32, 32)
        x = self.deocder_upsample1(x)              # Shape: (B, 128, 64, 64)
        x = self.deocder_layer2(x)                 # Shape: (B, 128, 64, 64)
        x = self.deocder_upsample2(x)              # Shape: (B, 128, 128, 128)
        x = self.deocder_layer3(x)                 # Shape: (B, 256, 128, 128)
        x = self.deocder_upsample3(x)              # Shape: (B, 256, 256, 256)
        x = self.deocder_layer4(x)                 # Shape: (B, 3, 256, 256)
        x = self.tanh(x)  

        

        # x = self.final_conv(x)
        print(f"shape of encoder output after final layer : {x.shape} -----------")

        return x

