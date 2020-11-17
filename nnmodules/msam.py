# from typing import Optional
import torch
import torch.nn as nn
from torch.nn.functional import relu, interpolate

from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock
from.unet import UNet3D


CHANNELS_DIMENSION = 1


class MSAM3D(nn.Module):
    """
    3D implementation of the Multi-modality Spatial Attention Module architecture (Fu et al., 2020)

    Note: "MSAM" here refers to the entire network (backbone + attention module), 
    unlike in the paper where it refers to only the attention module. 
    """

    def __init__(self, attention_module_config, backbone_config, output_attention_map=False):
        super().__init__()

        self.nn_name = 'msam3d'
        self.attention_module_config = attention_module_config
        self.backbone_config = backbone_config
        self.output_attention_map = output_attention_map

        # Attention module
        self.attention_module = UNet3D(**attention_module_config)
        
        # Backbone
        self.backbone = Backbone(backbone_config)    

    def forward(self, input_patches):
        PET, CT = input_patches

        # Run the attention module
        full_scale_map = self.attention_module(PET)
        full_scale_map = relu(full_scale_map)   # Shape: (N,1,D,H,W)
        
        # Create a list of downsampled versions of the attention map
        half_scale_map = downscale_by_two(full_scale_map)
        quarter_scale_map = downscale_by_two(half_scale_map)
        scaled_attention_maps = [full_scale_map, half_scale_map, quarter_scale_map]

        # Run the backbone network
        pred = self.backbone(CT, scaled_attention_maps)
        
        if self.output_attention_map:
            full_scale_map = torch.cat((1-full_scale_map, full_scale_map), dim=1)  # Make the channel number to 2, so that the patch aggregator can handle this
            return pred, full_scale_map
        else:
            return pred, None



class Backbone(nn.Module):
    """
    3D U-Net based backbone network for the MSAM
    """

    def __init__(self, backbone_config):
        super().__init__()

        # Backbone stuff --
        depth = backbone_config['num_encoding_blocks'] - 1

        # Force padding if residual blocks
        if backbone_config['residual']:
            padding = 1

        # Backbone encoder
        self.encoder = Encoder(
                                in_channels=backbone_config['in_channels'],
                                out_channels_first=backbone_config['out_channels_first_layer'],
                                dimensions=3,
                                pooling_type='max',
                                num_encoding_blocks=depth,
                                normalization=backbone_config['normalization'],
                                preactivation=False,
                                residual=backbone_config['residual'],
                                padding=padding,
                                padding_mode='zeros',
                                activation='ReLU',
                                initial_dilation=None,
                                dropout=0,
                                )

        # Backbone bottom (last encoding block)
        self.bottom_block = EncodingBlock(
                                        in_channels=self.encoder.out_channels,
                                        out_channels_first=self.encoder.out_channels,
                                        dimensions=3,
                                        normalization='batch',
                                        pooling_type=None,
                                        preactivation=None,
                                        residual=backbone_config['residual'],
                                        padding=padding,
                                        padding_mode='zeros',
                                        activation='ReLU',
                                        dilation=self.encoder.dilation,
                                        dropout=0,
                                    )

        # Backbone decoder
        power = depth
        in_channels_skip_connection = backbone_config['out_channels_first_layer'] * 2**power
        self.decoder = Decoder(
                                in_channels_skip_connection,
                                dimensions=3,
                                upsampling_type='conv',
                                num_decoding_blocks=depth,
                                normalization=backbone_config['normalization'],
                                preactivation=False,
                                residual=backbone_config['residual'],
                                padding=padding,
                                padding_mode='zeros',
                                activation='ReLU',
                                initial_dilation=self.encoder.dilation,
                                dropout=0,
                                )

        # Backbone classifier block
        in_channels = 2 * backbone_config['out_channels_first_layer']
        self.classifier = ConvolutionalBlock(
                                             dimensions=3, 
                                             in_channels=in_channels, 
                                             out_channels=2,
                                             kernel_size=1, 
                                             activation=None,
                                            )


    def forward(self, CT, scaled_attention_maps):

        # Run the encoder
        skip_connections, encoding = self.encoder(CT)
        encoding = self.bottom_block(encoding)

        # Gate the skip connections with the scaled attention maps
        gated_skip_connections = []
        for i, skip_connection in enumerate(skip_connections):            
            # Repeat the spatial map along the channels
            attention_map = scaled_attention_maps[i]
            n_channels = skip_connection.shape[CHANNELS_DIMENSION]
            spatial_gate = torch.repeat_interleave(attention_map, n_channels, dim=CHANNELS_DIMENSION) 

            gated_skip_connection =  spatial_gate * skip_connection
            gated_skip_connections.append(gated_skip_connection)

        # Run the decoder
        x = self.decoder(gated_skip_connections, encoding)
        return self.classifier(x)                                        



def downscale_by_two(input_attention_map):
    downscaled_attention_map = interpolate(input_attention_map, 
                                           scale_factor=(0.5, 0.5, 0.5), 
                                           mode='trilinear',
                                           align_corners=False)
    return downscaled_attention_map