#import importlib
#
# usage: python makeVariations.py -i /mydata/oreilly.pinned.edges -o /mydata/boreillytest -s 100 -d 3 -n 5 -a .7
#
##########  param parsing
import argparse
import sys

########## style transfer
#from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import librosa
from IPython.display import Audio, display
from PIL import Image

import scipy.stats as stats

import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.modules.module import _addindent

import copy
import os
from os import listdir
from os.path import isfile, join


import math

#for random feature projection
from scipy.stats import ortho_group
import soundfile as sf

import collections as c



if __name__ == "__main__":

    # allows str2bool as a type for parser
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Generate texture variations with different noise inits', add_help=False)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--idir', help="Path to input sound folder", required=True,
                        type=str, dest="idir")
    requiredNamed.add_argument('-o', '--outsounddir', help="Path to output sound folder", required=True,
                        type=str, dest="outsounddir")
    requiredNamed.add_argument('-s', '--steps', help="Steps for style transer", required=True,
                        type=int, dest="steps")

    parser.add_argument('-m', '--mdir', help="Path to model's folder", required=False,
                        type=str, dest="mdir", default="None")
    parser.add_argument('-n', '--numVariations', help='Number of variations for each input sound (default 1)', required=False,
                        type=int, dest="numVariations", default=1)
    parser.add_argument('-a', '--ampScale', help='amp scaling, (default=1)', required=False,
                        type=float, dest="ampScale", default=1.0)
    parser.add_argument('-d', '--dur', help='duration of sounds in seconds (default same as input sounds)', required=False,
                        type=float, dest="dur", default=2.0)
    parser.add_argument('-v', '--verbose', help='verbose run with printouts all over the place', required=False,
                        type=str2bool, dest="verbose", default=False)

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()
    print(f"Args are {args} and unkonwns are {unknown}")

#############################################################################################
###########   Internal Parameters ##############
# True uses PGHI (Truncated Gaussian window, log mag, PGHI reconstruction)
# False uses Log Mag spectrum and Griffin Lim reconstruction
tifresi=False

#Go up to 20000 for high quality if you are on a GPU
num_steps=args.steps
print(f'{num_steps=}')


learning_Rate= 1 #0.1 #.1  # must be smaller for more streams (if numStream=1, then learning_Rate can be 1)

variations = args.numVariations # no. of separate variations for a parameter setting

#############################################################################################

#librosa audio params
N_FFT = 512 
K_HOP = 128 
noiseframes=int(K_HOP*args.dur)

# architecture
"""use a custom convolutional network randomly initialized""" 
use01scale = False #set to true if want to scale img to [0,1] prior to training. Recommended if using VGG19
boundopt = False #whether to regularize the input within [lower,upper]. Recommended if using VGG19
whichChannel = "freq" #2d=2d conv, 1d options:freq=freq bins as channels, time= time bins as channels 
N_FILTERS = 512# 4096 #no. of filters in 1st conv layer
# hor_filter = 5 #width of conv filter, for 2d also the height of (square) kernel


#MS Number of separate CNNs operating on input

possible_kernels = [2,2,4,4,8,8,16,16] #32,64,128,256,512,1024,2048]
numStreams=len(possible_kernels)

# possible_kernels = [2,4,8,16,32,64,128,256,512,1024,2048]
# possible_kernels = [5,5,5,5,5,5]

hor_filters = [0]*numStreams
for j in range(numStreams):
    hor_filters[j]=possible_kernels[j]

#############################################################################################
def log_scale(img):
    img = np.log1p(img)
    return img

def inv_log(img):
    img = np.exp(img) - 1.
    return img

#############################################################################################
from tifresi.hparams import HParams
from tifresi.stft import GaussTruncTF

# from tifresi.transforms import log_spectrogram
# from tifresi.transforms import inv_log_spectrogram
#NOTE: Not using Marifioties 10 log_10 transform. Instead use natural log.
# This makes a HUGE difference !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
log_spectrogram=log_scale
inv_log_spectrogram=inv_log

# not sure HParams are being set properly here
HParams.stft_channels = N_FFT 
stft_channels = HParams.stft_channels 
HParams.hop_size  = K_HOP
hop_size =  HParams.hop_size
print(f'if tifresi: {hop_size=}')
HParams.sr=16000 

# empirically set: - too small, get low-res ringing; too high, get distortion
tfresiMagSpectScale=1 # Takes the [0,1] mag spectrogram and maps it to [0, tfresiMagSpectScale]

# For faster processin, a truncated window can be used instead
stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)

############################################################################################

use_cuda = torch.cuda.is_available() #use GPU if available
print('GPU available =',use_cuda)
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def read_audio_spectum(filename, tifresi=True):
    #x, fs = librosa.load(filename) #x=audiodata, fs=samplerate
    x, fs  = sf.read(filename)
    print(f'read_audio_spectum: input file sample rate = {fs}')
    print(f'read_audio_spectum: {len(x)=}')


#     x=x+np.random.normal(scale=.0001, size=N_SAMPLES)
    
    if tifresi :
        #x=np.append(x, np.zeros((hop_size-np.mod(len(x), hop_size))))
        print(f'np.mod(len(x), stft_channels) = {np.mod(len(x), stft_channels)}')
        x=np.append(x, np.zeros(stft_channels-np.mod(len(x), stft_channels)))
        print(f'{len(x)=}, {np.mod(len(x), hop_size)=}')
        print(f' number of hops is {len(x)//hop_size}')
    
    #N_SAMPLES = len(x)
    if tifresi :
        print(f'In read audio, doing tifresi spectrogram')
#         R=tfresiMagSpectScale*stft_system.spectrogram(x)
        R=stft_system.spectrogram(x)
    else :
        print(f'In read audio, doing librosa spectrogram')
        R = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=K_HOP, win_length=N_FFT,  center=False))    
        
    print(f'R range is  [{np.amin(R)}, {np.amax(R)}')
    print(f'K_HOP - {K_HOP} and N_FFT is {N_FFT}')
    print(f'R shape is {R.shape}')
    return R, fs


def findMinMax(img):
    return int(math.floor(np.amin(img))),int(math.ceil(np.amax(img)))

def img_scale(img,datasetMin,datasetMax,scaleMin,scaleMax):
    """scales input numpy array from [datasetMin,datasetMax] -> [scaleMin,scaleMax]"""    
    shift = (scaleMax-scaleMin) / (datasetMax-datasetMin)
    scaled_values = shift * (img-datasetMin) + scaleMin
    print("img_scale: Using [{0},{1}] -> [{2},{3}] for scale conversion".format(datasetMin,datasetMax,scaleMin,scaleMax))
    return scaled_values

def img_invscale(img,datasetMin,datasetMax,scaleMin,scaleMax):
    """scales input numpy array from [scaleMin,scaleMax] -> [datasetMin,datasetMax]"""
    shift = (datasetMax-datasetMin) / (scaleMax-scaleMin)
    scaled_values = shift * (img-scaleMin) + datasetMin
    print("img_invscale: Using [{0},{1}] -> [{2},{3}] for inverse scale conversion".format(scaleMin,scaleMax,datasetMin,datasetMax))
    return scaled_values
    
#if 0
    # use below functions to use librosa db scale, normalized to [0,1]
    # note that this scaling does not work well for style transfer
    def db_scale(img,scale=80):
        img = librosa.amplitude_to_db(img)
        shift = float(np.amax(img))
        img = img - shift #put max at 0
        img = img/scale #scale from [-80,0] to [-1,0]
        img = img + 1. #shift to [0,1]
        img = np.maximum(img, 0) #clip anything below 0
        return img, shift

    def inv_db(img,shift,scale=80):
        img = img - 1. #shift from [0,1] to [-1,0]
        img = img * scale #[-1,0] -> [-80,0]
        img = img + shift
        img = librosa.db_to_amplitude(img)    
        return img


##################################

# custom weights initialization
def weights_init(m,hor_filter):
    std = np.sqrt(2) * np.sqrt(2.0 / ((N_FREQ + N_FILTERS) * hor_filter))
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)
        m.weight.data.normal_(0.0, std)

class style_net(nn.Module):
    """Here create the network you want to use by adding/removing layers in nn.Sequential"""
    def __init__(self,hor_filter):
        super(style_net, self).__init__()
#         self.hor_filter=hor_filter
        self.layers = nn.Sequential(c.OrderedDict([
                            ('conv1',nn.Conv2d(IN_CHANNELS,N_FILTERS,kernel_size=(1,hor_filter), padding='valid', padding_mode='circular',bias=False)),
                            ('relu1',nn.ReLU())#,
#                             ('max1', nn.MaxPool2d(kernel_size=(1,2))),  # if stacking more conv layers...
            
        ]))

    def forward(self,input):
        out = self.layers(input)
        return out


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size() #a=batch size(=1)
                                  #b=number of feature maps
                                  #(c,d)=dimensions of a feat. map (N=c*d) -> for 1D conv c=1
        #features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        features = input.view(b, a * c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight, layer): #weight here is the alpha tuning (how much content vs style)
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss(size_average=False)
        #print(f'Creating StyleLoss module - this one for layer {layer} with target shape {target.shape}' )


    def forward(self, input):
        #print(f' StyleLoss.forward with input of shape {input.shape}')
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target) #/sum(sum(self.target**2)) #target=gram mat for style img, G=gram mat for current input ie. noise
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_graph=True)
        return self.loss


###################################
#rebuild network with the layers we want - do this for every variation
#   the cnn is already built and initialized before calling this function
def get_style_model_and_losses(cnn, style_img, content_img=None,
                               style_weight=1, content_weight=0,
                               content_layers=[],
                               style_layers=[], style_img2=None):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []
    
    model = nn.Sequential()
    layer_list = list(cnn.layers)
    
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    #here we rebuild the network adding the in content and style loss "layers"   
    i = 1  
    for layer in layer_list:

        
        if isinstance(layer, nn.Conv2d): #if layer in vgg19 belong to class nn.Conv2d
            name = "conv_" + str(i)
            model.add_module(name, layer) #add that layer to our sequential model
            
            if content_img != None:
                if name in content_layers: #at the right depth add the content loss "layer"
                    # add content loss:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

            if name in style_layers: #at the right depth add the content loss "layer"
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight, name)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU): #do the same for ReLUs
            name = "relu_" + str(i)
            model.add_module(name, layer)
            
            if content_img != None:
                if name in content_layers:
                    # add content loss:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                
                
                if 0 : # Replace target with Random Symmetric matrix by shuffling
                    
                    #test to see if random symmetric Gram Matrix can be matched with style transfer iteration
                    # randomly shuffle Gram stack
                    print(f'(Before) TARGET SYMMETRY? {((target_feature_gram.transpose(1, 2) == target_feature_gram).all())=}')
                    
                    #shuffle
                    idx = torch.randperm(target_feature_gram.nelement())
                    target_feature_gram = target_feature_gram.view(-1)[idx].view(target_feature_gram.size())
                    print(f'{target_feature_gram.size()=}')
                    #symmeterize
                    target_feature_gram=(target_feature_gram + target_feature_gram.transpose(1, 2))/2

                    print(f'(After) TARGET SYMMETRY? {((target_feature_gram.transpose(1, 2) == target_feature_gram).all())=}')
                
                
                style_loss = StyleLoss(target_feature_gram, style_weight, name)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss) 
                
            i += 1

        if isinstance(layer, nn.MaxPool2d): #do the same for maxpool
            name = "pool_" + str(i)
            model.add_module(name, layer)
            
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight, name)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
            
            #avgpool = nn.AvgPool2d(kernel_size=(1,2),
            #                stride=layer.stride, padding = layer.padding)
            #model.add_module(name, avgpool)  # *** can also replace certain layers if we want eg. maxpool -> avgpool


    #for param in model.parameters():
    #    param.requires_grad = False
    return model, style_losses, content_losses

###############################################
"""image to input in generative network"""
# variations is a global - the number of different new sounds to create for each input sound.  Ycuk.

def in_img(n_freqs, nsamples):
    """initialize (variations) number of unique random noise input images"""  
    input_imgs = []
    #rand_tensor = torch.randn(1,n_freqs,1,nsamples)*1e-3
#     lower,upper,mu,sigma = 1e-3,1,.5,1e-1
#     lower,upper,mu,sigma = -50,0,-45,7
    lower,upper,mu,sigma = 1e-9,0.7,0.001,0.01

    dist = stats.truncnorm((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)
    
    i = 0
    while i < variations:
         #[batch,channels,h,w]
        if whichChannel == "2d":
            samples = dist.rvs([1,1,n_freqs,nsamples])
            input_imgs.append(Variable(torch.from_numpy(samples)).type(dtype))
            #input_imgs.append(Variable(torch.rand(1,1,n_freqs,nsamples)*1e-3).type(dtype))
            #input_imgs.append(Variable(torch.randn(1,1,n_freqs,nsamples)*1e-3).type(dtype))
        if whichChannel == "freq":
            #samples = dist.rvs([1,n_freqs,1,nsamples])
            #input_imgs.append(Variable(torch.from_numpy(samples)).type(dtype))
            #input_imgs.append(Variable(torch.rand(1,n_freqs,1,nsamples)*1e-3).type(dtype))
            input_imgs.append(Variable(mu+torch.randn(1,n_freqs,1,nsamples)*1e-2).type(dtype))
        elif whichChannel == "time":
            samples = dist.rvs([1,nsamples,1,n_freqs])
            input_imgs.append(Variable(torch.from_numpy(samples)).type(dtype))
            #input_imgs.append(Variable(torch.rand(1,nsamples,1,n_freqs)*1e-3).type(dtype))
            #input_imgs.append(Variable(torch.randn(1,nsamples,1,n_freqs)*1e-3).type(dtype))
        i +=1
    return input_imgs


# just make one - we don't run these in batches anyway - why do we need an array of them??
def in_img1(n_freqs, nframes):
    """generate unique random noise input images"""  

    lower,upper,mu,sigma = 1e-9,0.7,0.001,0.01

    dist = stats.truncnorm((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)
    

    if whichChannel == "2d":
        samples = dist.rvs([1,1,n_freqs,nframes])
        input_img = Variable(torch.from_numpy(samples)).type(dtype)
    if whichChannel == "freq":
        input_img = Variable(mu+torch.randn(1,n_freqs,1,nframes)*1e-2).type(dtype)
    elif whichChannel == "time":
        samples = dist.rvs([1,nframes,1,n_freqs])
        input_img = Variable(torch.from_numpy(samples)).type(dtype)


    return input_img

#######################

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

######################
# Run for a single variation

def run_style_transfer(cnnlist, style_img, input_img, content_img=None, num_steps=num_steps,
                       style_weight=1, content_weight=0, reg_weight=0.01): 
    """Run the style transfer"""

    #MS - one model, loss accumulator, and optimizer per stream 
    modelMS=[None for j in range(numStreams)]
    style_lossesMS=[None for j in range(numStreams)] 
    content_lossesMS=[None for j in range(numStreams)]
#     optimizerMS=[None for j in range(numStreams)]

    #MS one input_param for all streams, each operating on the same data. 
    input_param = nn.Parameter(input_img.data)
    
    
    prev = input_param.data
    
    # first create the separate cnn models and losses
    for j in range(numStreams) :
        #print(f'Building the style transfer model for stream {j}..')
        modelMS[j], style_lossesMS[j], content_lossesMS[j] = get_style_model_and_losses(cnnlist[j],
            style_img, content_img, style_weight, content_weight, style_layers=style_layers_default)
        #input_paramMS[j], optimizerMS[j] = get_input_param_optimizer(input_img)
        #print(f'CREATING, style_lossesMS is {style_lossesMS}')
        #input_param is the same variable passed to each separate CNN
        #MS Changed learning rate to prevent instabilities with lr=1.
#         optimizerMS[j] = optim.LBFGS([input_param], lr=learning_Rate)

        if args.verbose :
            print("Input range:",torch.max(input_param.data),torch.min(input_param.data))   
            print(modelMS[j])
            print('Optimizing..')
    optimizerMS = optim.LBFGS([input_param],lr=learning_Rate)
        
    
    print(f'Created {j+1} MS models')
    # print(f' input_param[0] is input_param[1] ? ...... {input_paramMS[0] is input_paramMS[1]}')    # FALSE!    
    

    step = [0]
    
    if boundopt == True:
        bound = OptReg(1,0.005, reg_weight)
    

    #plt.figure()
    while step[0] <= num_steps:

        step[0] += 1 
        #print("Current step Number = ",step[0])
        
        #MS create the closure, and then run the optimizer for each stream in sequence
        #print("numStreams = ",numStreams)
        def closure():
            optimizerMS.zero_grad()
            style_score = 0
            content_score = 0
            for j in range(numStreams) :
        
            
                # correct the values of updated input image
                #if use01scale == True:
                    #input_param.data.clamp_(0, 1)

                #print("Current stream Number in closure() = ",j)
#                 optimizerMS.zero_grad()
                modelMS[j](input_param)
                

                if boundopt == True:
                    bound.zero_grad()
                    reg = bound(input_param)
                    if not torch.equal(reg.data,torch.cuda.FloatTensor([0])):
                        reg.backward()

                for sl in style_lossesMS[j]:
                    ####mapweight= map(step[0],progressive_proportion*num_steps*(1-(j+1)/numStreams))
                    ####style_score += mapweight*sl.backward() #call backward method to grab the loss
                    #print(f'map step={step[0]}, stream={j} = {mapweight}')
                    style_score += sl.backward() 
                
                #print("stylescore for run", step[0], " stream ",j ," is ",style_score)

                #for cl in content_losses:
                #    content_score += cl.backward()

                if boundopt == True:
                    total_loss = style_score + reg  #+ content_score
                else:
                    total_loss = style_score

            return total_loss/numStreams   #return value for closer()

            # The update to an image happens in here:
        closureloss=optimizerMS.step(closure)  #optimizer.step will call closure with its closure vars; #executes 20 iterations in this one line
        if torch.isnan(closureloss) :
            break
        if step[0] % 10 == 0:
            print(f'step: {step}, total_loss/numStreams : {closureloss.data:4f}')
      
        # END MS loop
        
        
        
    # a last correction...
    #if use01scale == True:
        #input_param.data.clamp_(0, 1)
    #plt.show()
    return input_param.data, closureloss   # Can return any of the MS input_params - they all refer to the same object


####################################
def tensorSpect2np(tensor, n_freqs, n_frames, channels=1) : 
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    if whichChannel == "2d":
        image = image
    if whichChannel == "freq":
        image = image.permute(0,2,1,3).contiguous() # get the dimensions in proper order
    elif whichChannel == "time":
        image = image.permute(0,2,3,1).contiguous() # get the dimensions in proper order
    image = image.view(n_freqs, n_frames)  # remove the fake batch dimension
    
    return image.numpy() #convert pytorch tensor to numpy array

#####################################################################################
#####################################################################################
# LOOP over all sounds in a directory
filelist = [f for f in listdir(args.idir) if f.endswith('.wav')]
print(f"filelist is {filelist} ")
insoundCount=0
for insound in filelist :
    STYLE_FILENAME=join(args.idir,insound)
    print(f"target sound is {STYLE_FILENAME}")


    R, fs = read_audio_spectum(STYLE_FILENAME, tifresi)
    print(f"raw spectrogram R range before log and scaling: [{np.amin(R)},{np.amax(R)}] ")

    if tifresi :
        print('TIFResi: log_spectrogram')
        a_style = log_spectrogram(R)
    
    else :
        print('HUZ: log_scase')
        a_style = log_scale(R)
        
    print(f"    LOG    range before scaling: [{np.amin(a_style)},{np.amax(a_style)}]",)
    print(f'shape of a_style is {a_style.shape}')

    if use01scale == True:
        a_min,a_max = findMinMax(a_style)
        a_style = img_scale(a_style,a_min,a_max,0,1)

    ##############################################################
    # Get the input spectrogram as a pytorch variable
    N_FRAMES = a_style.shape[1] #time bins
    N_FREQ = a_style.shape[0] #freq bins

    if whichChannel == "2d":
        IN_CHANNELS = 1
    elif whichChannel == "freq":
        IN_CHANNELS = N_FREQ
    elif whichChannel == "time":
        IN_CHANNELS = N_FRAMES
    

    a_style = np.ascontiguousarray(a_style[None,None,:,:]) #[batch,channels,freq,samples]
    if whichChannel == "2d":
        a_style = torch.from_numpy(a_style) #pytorch:[batch,channels(1),height(freq),width(samples)]
    elif whichChannel == "freq":
        a_style = torch.from_numpy(a_style).permute(0,2,1,3) #pytorch:[batch,channels(freq),height(1),width(samples)]
    elif whichChannel == "time":
        a_style = torch.from_numpy(a_style).permute(0,3,1,2) #pytorch:[batch,channels(samples),height(1),width(freq)]

    style_img = Variable(a_style).type(dtype) #convert to pytorch variable
    print("Using whichChannel ==",whichChannel," Input shape:",style_img.data.shape)



    ###############    make the ensemble of CNNs  ########################################
    cnnlist=[] 
    #MS create a separate CNN for each stream
    for j in range(numStreams) :
        cnn = style_net(hor_filters[j])
        cnn.apply(lambda x, f=hor_filters[j]: weights_init(x,f))
        for param in cnn.parameters():
            param.requires_grad = False
        print(list(cnn.layers))

        # move it to the GPU if possible:
        if use_cuda:
            cnn = cnn.cuda()
        
        cnnlist.append(cnn)

    # Add the style/content loss 'layer' after the specified layer:
    content_layers_default = [] #ignore for now
    # style_layers_default = ['relu_1']
    style_layers_default = ['relu_1']




    ##########################################################
    ####   Now, run the "inner loop" over each variation we want to generate

    # Next create array of noise images, one for each variation we will generate
    #input_imgs = in_img1(N_FREQ, noiseframes) # 
    #print(f'{input_imgs[i].shape=}')

    outputs = []   #reinitialized for each input sound
    i = 0
    while i < variations:

        print(f' VARIATION # {i}')
        strikes=0 # count's successive times the loss fails to converge - this is a rare event and apparently dependent on a bad input noise choice
        if strikes < 3 :
            input_img = in_img1(N_FREQ, noiseframes)
            output, loss = run_style_transfer(cnnlist, style_img, input_img)
            if torch.isnan(loss) :
                strikes=strikes+1
                print(f'!! Loss for variation {i} is NaN - will try another input initialization !!')
                continue
        else :
            print(f'!! Loss for variation {i} is NaN - GIVING UP AFTER THREE DIFFERENT INUT ATTEMPTS !!')

        print(f'variation {i} output shape is {output.data.shape}')

        #plt.figure().set_size_inches(20, 20)
        out = tensorSpect2np(output, N_FREQ, noiseframes)
        print(f'variation {i} output numpy shape is {out.shape}')
        #out = imshow(output, N_FREQ, int(N_FRAMES*SYNTHETIC_LENGTH_FACTOR),title='Output Image')
        outputs.append(out)

        i=i+1


    print(f' ---------------   Now do the inversion and output of all variations for this sound ---------')

########################################################################
########################################################################

# First inverse log for all images 

    if tifresi :
        #### Shift log spec to use max range before thresholding for tfresi inversion
        #outputs=[x+1-np.amax(x) for x in outputs]
        print('TIFResi: inv_log_spectrogram')
        if use01scale == True:
            foo = [img_invscale(x,a_min,a_max,0,1) for x in outputs]
            
            out_spec = [inv_log_spectrogram(x+SPECTOFFSET) for x in foo]

            for j in range(len(outputs)):
                print(f'   RESCALED: Min/Max of outputs:  {np.amin(foo[j])},{np.amax(foo[j])}') # check values make sense

        else:
            out_spec = [inv_log_spectrogram(x+SPECTOFFSET) for x in outputs] 

            #out_spec = [inv_log(x+SPECTOFFSET) for x in outputs]
            for j in range(len(outputs)):
                print(f'   inv_log_spectrogram: Min/Max of outputs:  {np.amin(out_spec[j])},{np.amax(out_spec[j])}') # check values make sense

    else : 
        print('HUZ: inv_log')
        if use01scale == True:
            out_spec = [inv_log(img_invscale(x,a_min,a_max,0,1)) for x in outputs]
        else:
            out_spec = [inv_log(x) for x in outputs]

        for j in range(len(out_spec)):
            print(f'Min/Max of out_spec: {np.amin(out_spec[j])}, {np.amax(out_spec[j])}') # check values make sense
              
            
    ########
    print(f'Now invert the spectrograms of the variations to audio')
    for index,value in enumerate(out_spec):
        print(f'shape of value is {value.shape}')
        
        if tifresi :
            #print(f'TIFResi invert mag to audio')
            #print("shape of mag = ",np.shape(value))
            S = value
            bar=S#-np.amin(S)+.0001
    #         bar = np.abs(bar)
            #bar=bar/np.amax(bar)
            bar = np.clip(bar, 0,None)
            if verbose :
                print(f'Min/Max of spectrogram: {np.amin(bar)}, {np.amax(bar)}') # check values make sense
            x = stft_system.invert_spectrogram(bar) 
        else :
            #print(f'Huz invert mag to audio')
            p = 2 * np.pi * np.random.random_sample(value.shape) - np.pi #start with some phase
            #Griffin Lim
            for i in range(50):
                S = value * np.exp(1j*p) #use magnitude a given by spectrogram and some random phase
                x = librosa.istft(S,hop_length=K_HOP, win_length=N_FFT, center=False) #do the inverse transform
                p = np.angle(librosa.stft(x, n_fft=N_FFT, hop_length=K_HOP, win_length=N_FFT, center=False)) #use this new phase value

        try:
            os.makedirs(args.outsounddir)
        except FileExistsError:
            # directory already exists
            pass



        base=os.path.basename(STYLE_FILENAME)
        OUTPUT_FILENAME=args.outsounddir + "/" +  os.path.splitext(base)[0] + '_gv.' + str(index) + '.wav'
        sf.write(OUTPUT_FILENAME, args.ampScale*x, fs)    


    print(f'++++++++++++++++++++++++++ Done with insound {insoundCount} Next input sound ++++++++++++++++++++++++++++++')
    insoundCount=insoundCount+1






    

