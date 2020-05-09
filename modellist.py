from model import Unet, fcn_resnet

class Modellist():
    def __init__(self):
        print('modelnum list')
        print('-'*30)
        print('1: Unet')
        print('2: fcn_resnet')
        print('3: x')
        print('-'*30)

    def __call__(self,x,seon):
        return {1: Unet.Unet(1),
        2: fcn_resnet.fcn_resnet.FCN()
        #3: Vgg.VGG16(seon)
        }
