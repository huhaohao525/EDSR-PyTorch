import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import cv2
import numpy as np

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

types_set = set()

big_canvas = np.zeros((1280 * 2, 480), np.uint8)

def visOneKernel(ker):
    pass
    # inp: numpy
    print('before ker:', ker)
    big_ker = cv2.resize(ker, (ker.shape[0]*5, ker.shape[1]*5), interpolation=cv2.INTER_NEAREST)
    big_ker = ((big_ker + 0.32) * 255).astype(np.uint8)
    print('after ker:', big_ker)
    #cv2.imwrite('/data1/tanhao.hu/tmp/kernel.png', big_ker)
    return big_ker

y = 0
x = 0

def paintOneKernel(ker, idx):
    pass
    global x, y
    try:
        big_canvas[y : y + 15, x : x + 15] = ker
        x += 16
        if x > 320:
            x = 0
            y += 16
    except:
        print('failed one, kernel shape is :', ker.shape)
        pass



def checkModuleTypes(mod):
    print(type(mod))
    if hasattr(mod, 'weight'):
        if mod.weight.data.cpu().numpy().shape[0] == 64:
            for k in range(64):
                #print(mod.weight.data.cpu().numpy().shape)
                patch = visOneKernel(mod.weight.data.cpu().numpy()[k][0])
                paintOneKernel(patch, k)
            
        print('shape is ', mod.weight.shape, 'bias is ', mod.bias.shape)
        print('type of data.item() is ', type(mod.weight.data.cpu().numpy()))
    types_set.add(str(type(mod)))
    if hasattr(mod, '_modules'):
        for k, v in mod._modules.items():
            print('k is : {}'.format(k))
            checkModuleTypes(v)
    else:
        #print(type(mod))
        pass

def forwardAll(mod, inp):
    pass
    if hasattr(mod, 'forward'):
        print('has forward')
    else:
        print('no forward')

    if hasattr(mod, '_modules'):
        for k, v in mod._modules.items():
            print('k is : {}'.format(k))
            forwardAll(v, inp)
    

'''def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:'''

if checkpoint.ok:
    loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    head = _model._modules['model']._modules['head']._modules
    print(head['0'].weight.data.max())
    print(head['0'].weight.data.min())
    print(type(head['0']))
    for k in head:
        print("k is {}".format(k))

    '''
    for k in _model._modules['model']['EDSR']: #_model._modules:
        #print("k is : {}, v is : {}".format(k, v))
        print(k)
    '''

    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    '''
    t = Trainer(args, loader, _model, _loss, checkpoint)    
    while not t.terminate():
        t.train()
        t.test()
    '''

    checkModuleTypes(_model)
    cv2.imwrite('/data1/tanhao.hu/tmp/kernel.png', big_canvas)
    forwardAll(_model, None)
    
    print(types_set)

    checkpoint.done()



if __name__ == '__main__':
    pass
