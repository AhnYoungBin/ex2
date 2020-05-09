from model import Unet, fcn_resnet

class Modellist():
    def __init__(self):
        print('modelnum list')
        print('-'*100)
        print('1: Unet')
        print('2: fcn_resnet')
        print('3: x')
        print('-'*100)

    def __call__(self,x):
        return {1: Unet.UNet(1),
        2: fcn_resnet.FCN()
        #3: Vgg.VGG16(seon)
        }[x]
