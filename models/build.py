# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# from .fsa_swin import FSASwinTransformer
from .fsa_deit import fsa_deit_tiny, fsa_deit_small, fsa_deit_base
from .fsa_pvt import fsa_pvt_tiny, fsa_pvt_small, fsa_pvt_medium, fsa_pvt_large
# from .fsa_cswin import FSA_CSWin_64_24181_tiny_224, FSA_CSWin_96_36292_base_224, \
#     FSA_CSWin_96_36292_base_384, FSA_CSWin_64_36292_small_224


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'fsa_swin':
        model = FSASwinTransformer(img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                     in_chans=config.MODEL.SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                     depths=config.MODEL.SWIN.DEPTHS,
                                     num_heads=config.MODEL.SWIN.NUM_HEADS,
                                     window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.SWIN.APE,
                                     patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     attn_type=config.MODEL.FSA.ATTN_TYPE)

    elif model_type in ['fsa_deit_tiny', 'fsa_deit_small', 'fsa_deit_base',
                        'fsa_deit_base_d21', 'fsa_deit_mini_2x', 'fsa_deit_tiny_2x',
                        'fsa_deit_tiny_4x', 'fsa_deit_small_2x',
                        'fsa_deit_base_fsa2', 'fsa_deit_base_fsa4', 'fsa_deit_base_fsa6']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,)')

    elif model_type in ['fsa_pvt_tiny', 'fsa_pvt_small', 'fsa_pvt_medium', 'fsa_pvt_large',
                        'fsa_pvt_tiny_2x']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.FSA.ATTN_TYPE,'
                                  'fsa_sr_ratios=str(config.MODEL.FSA.PVT_LA_SR_RATIOS))')

    elif model_type in ['FSA_CSWin_64_24181_tiny_224', 'FSA_CSWin_64_24322_small_224',
                        'FSA_CSWin_96_36292_base_224', 'FSA_CSWin_64_36292_small_224',
                        'FSA_CSWin_96_36292_base_384']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.FSA.ATTN_TYPE,'
                                  'la_split_size=config.MODEL.FSA.CSWIN_LA_SPLIT_SIZE)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
