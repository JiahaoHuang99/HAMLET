import functools
import torch
import torchvision.models
from torch.nn import init
import numpy as np


def load_net(MODEL_NAME, net_params):
    """
    select the network
    """
    # Airfoil

    # Darcy flow
    if MODEL_NAME in ['DARCY_FLOW_M21a', 'DARCY_FLOW_M21b']:
        from nets.darcy_flow.DarcyFlowEncoderM21 import DarcyFlowEncoder as encoder
        from nets.darcy_flow.DarcyFlowDecoderM21 import DarcyFlowDecoder as decoder

    # Diffusion Reaction 2D
    elif MODEL_NAME in ['DIFFUSION_REACTION_2D_M21a', 'DIFFUSION_REACTION_2D_M21b']:
        from nets.diffusion_reaction_2d.DiffusionReaction2DEncoderM21 import DiffusionReaction2DEncoder as encoder
        from nets.diffusion_reaction_2d.DiffusionReaction2DDecoderM21 import DiffusionReaction2DDecoder as decoder

    # Shallow Water 2D
    elif MODEL_NAME in ['SHALLOW_WATER_2D_M21a', 'SHALLOW_WATER_2D_M21b']:
        from nets.shallow_water_2d.ShallowWater2DEncoderM21 import ShallowWater2DEncoder as encoder
        from nets.shallow_water_2d.ShallowWater2DDecoderM21 import ShallowWater2DDecoder as decoder

    else:
        raise ValueError('Network {} not found'.format(MODEL_NAME))

    enc = encoder(net_params)
    dec = decoder(net_params)

    init_weights(enc,
                 init_type=net_params['init_type'],
                 init_bn_type=net_params['init_bn_type'],
                 gain=net_params['init_gain'])

    init_weights(dec,
                 init_type=net_params['init_type'],
                 init_bn_type=net_params['init_bn_type'],
                 gain=net_params['init_gain'])

    return enc, dec


# --------------------------------------------
# weights initialization
# --------------------------------------------
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        EXCP_LIST = ['ConvectionDiffusionEncoder', 'ConvectionDiffusionDecoder', 'CrossLinearAttention', 'LinearAttention']
        EXCP = np.prod([classname.find(exp) == -1 for exp in EXCP_LIST])

        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and EXCP:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        # elif (classname.find('BatchNorm2d') != -1) or (classname.find('BatchNorm1d') != -1) or (classname.find('GroupNorm') != -1) or (classname.find('InstanceNorm1d') != -1):
        #     if init_bn_type == 'uniform':  # preferred
        #         if m.affine:
        #             init.uniform_(m.weight.data, 0.1, 1.0)
        #             init.constant_(m.bias.data, 0.0)
        #     elif init_bn_type == 'constant':
        #         if m.affine:
        #             init.constant_(m.weight.data, 1.0)
        #             init.constant_(m.bias.data, 0.0)
        #     else:
        #         raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))
        #
        # elif  (classname.find('LayerNorm') != -1):
        #     if init_bn_type == 'uniform':  # preferred
        #         if m.elementwise_affine:
        #             init.uniform_(m.weight.data, 0.1, 1.0)
        #             init.constant_(m.bias.data, 0.0)
        #     elif init_bn_type == 'constant':
        #         if m.elementwise_affine:
        #             init.constant_(m.weight.data, 1.0)
        #             init.constant_(m.bias.data, 0.0)
        #     else:
        #         raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))


    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')


