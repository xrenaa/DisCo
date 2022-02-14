import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optimizer

import itertools

import tqdm as tqdm
import argparse
import random
import json

from models.DisCo.latent_deformator import LatentDeformator
from models.loader import get_generator, get_encoders

from utils import *
from visualization import generate, visualize_GAN

parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--G", type=str, default="stylegan",
                    help="the type of generator: stylegan/sngan/biggan")
parser.add_argument("--dataset", type=int, default=0,
                    help="type of dataset")
parser.add_argument("--exp_name", type=str, default="train",
                    help="experiment name")
parser.add_argument("--index", type=int, default=0,
                    help="the index of pretrained generator: range from 0-5")
parser.add_argument("--z_dim", type=int, default=64,
                    help="the dimension of the output for the encoder")
parser.add_argument("--dim", type=int, default=64,
                    help="the number of directions used in the navigator")
parser.add_argument("--type", type=int, default= 0,
                    help="the type of navigator: 0 is orthogonal, 1 is projection")
parser.add_argument("--start", type=str, default="W",
                    help="the start space: W is W space in styleGAN2, Z is Z space in styleGAN2, S is S space in styleGAN2")
parser.add_argument("--end", type=str, default="V",
                    help="the start space: V is the variation space")
parser.add_argument("--B", type=int, default=32,
                    help="batch size")
parser.add_argument("--N", type=int, default=32,
                    help="the number of positive samples")
parser.add_argument("--K", type=int, default=64,
                    help="the number of negative samples")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="the learning rate") 
parser.add_argument("--max_iter", type=int, default=7e4,
                    help="the number of training iteration") 
parser.add_argument("--thresh", type=float, default=0.95,
                    help="the thresh hold for hard negative flipping")
parser.add_argument("--flipping", type=int, default=1,
                    help="whether to perfrom hard negative flipping")
parser.add_argument("--entropy", type=int, default=1,
                    help="whether to perfrom entropy-based loss")
args = parser.parse_args([])
    
max_iter = args.max_iter

batch_size = args.B
N = args.N
K = args.K

shift_scale = 6.0
min_shift = 0.5

# set random seed
seed = np.random.randint(1e6)
random_seed(seed)

# get the name of dataset
choices = ["shapes3d", "mpi3d", "cars3d", "color","noisy", "MNIST", "Anime"]
dataset = choices[args.dataset]
args.dataset = dataset

# get generators
generator, generator_latent_dim, nc = get_generator(args)
generator.eval().cuda()
for p in generator.parameters():
    p.requires_grad_(False)
    
# which space will the generator start
generator.type = args.G
generator.generator_latent_dim = generator_latent_dim

generator.Z = False
generator.W = False
generator.S = False

if args.G == "stylegan":
    if args.start == "W":
        generator.W = True
    elif args.start == "S":
        generator.S = True
    else:
        generator.Z = True
else:
    # for other types of generators, you only have the Z space (noise space)
    generator.Z = True

# get the navigator
used_dim = args.dim # use the first "used_dim" directions in the navigator

if args.type == 0:
    # the navigator is a orthogonal one
    total_dim = generator_latent_dim
    navigator = LatentDeformator( shift_dim= total_dim,
                    input_dim= total_dim,
                    out_dim= generator_latent_dim,
                    type=DEFORMATOR_TYPE_DICT["ortho"],
                    random_init= True).cuda()
if args.type == 1:
    # the navigator is a projection one
    total_dim = used_dim
    navigator = LatentDeformator( shift_dim= total_dim,
                    input_dim= total_dim,
                    out_dim= generator_latent_dim,
                    type=DEFORMATOR_TYPE_DICT["proj"],
                    random_init= True).cuda()    

navigator.train()

# get the encoder (Contrastor)
encoder = get_encoders(nc, args)
encoder.cuda()

# get the optim and loss
model_chain = itertools.chain(navigator.parameters(), encoder.parameters())
cross_entropy = nn.BCEWithLogitsLoss()
optim = optimizer.Adam(model_chain, lr=args.lr, betas=(0.9, 0.999))

# now init the logger path
encoder.model_name = "./experiments/%s/%s/" % (dataset, args.exp_name + str(args.index))

if not os.path.exists("./experiments/%s" % dataset):
    os.mkdir("./experiments/%s" % dataset)
    
if not os.path.exists("./experiments/%s/%s" % (dataset, args.exp_name+ str(args.index))):
    os.mkdir("./experiments/%s/%s" % (dataset, args.exp_name + str(args.index)))

if not os.path.exists(os.path.join(encoder.model_name, "viz")):
    os.mkdir(os.path.join(encoder.model_name, "viz"))
    
with open(os.path.join(encoder.model_name, "config.json"), 'w') as f:
    json.dump(vars(args),f)

avgs = MeanTracker('loss'), MeanTracker('logits_loss'),MeanTracker('entropy_loss')
avg_loss, avg_logits_loss, avg_entropy_loss = avgs

def make_specific_shift(target_indices, batch_size, latent_dim):
    target_indices = target_indices.repeat(N)
    shifts =  torch.randn(target_indices.shape, device='cuda')

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > min_shift) & (shifts < 0)] = -min_shift
    
    try:
        latent_dim[0]
        latent_dim = list(latent_dim)
    except Exception:
        latent_dim = [latent_dim]
    
    z_shift = torch.zeros([batch_size] + latent_dim, device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val
    
    return z_shift

def make_negative_shift(target_indice, batch_size, latent_dim, used_dim):
    
    r = [*range(0, target_indice), *range(target_indice+1, used_dim)]
    negative_indices = torch.randint(0, used_dim, [batch_size], device='cuda')
    
    shifts =  torch.randn(negative_indices.shape, device='cuda')

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > min_shift) & (shifts < 0)] = -min_shift
    
    try:
        latent_dim[0]
        latent_dim = list(latent_dim)
    except Exception:
        latent_dim = [latent_dim]
    
    z_shift = torch.zeros([batch_size] + latent_dim, device='cuda')
    for i, (index, val) in enumerate(zip(negative_indices, shifts)):
        if index == target_indice:
            index = random.choice(r)
        z_shift[i][index] += val
    
    return z_shift

out = False
global_iter = 0
pbar = tqdm.tqdm(total = max_iter)

while not out:
    global_iter += 1
    pbar.update(1)
    
    encoder.train()
    navigator.train()
    generator.zero_grad()
    
    target_indice = torch.randint(0, used_dim, [1], device='cuda') # the selected direction to be positive

    noise = torch.randn(batch_size, generator_latent_dim).cuda()
    imgs, z = generate(generator, noise, return_latent = True)

    shifts_1 = make_specific_shift(target_indice, batch_size, total_dim)
    shifts_1 = navigator(shifts_1)
    imgs_shifted_1 = generate(generator, z + shifts_1)

    noise_positive = torch.randn(N, generator_latent_dim).cuda()
    imgs_positive, z_positive = generate(generator, noise_positive, return_latent = True)

    shifts_2 = make_specific_shift(target_indice, N, total_dim)
    shifts_2 = navigator(shifts_2)
    imgs_shifted_2 = generate(generator, z_positive + shifts_2)

    imgs_feature = encoder((imgs+1) / 2).view(batch_size, -1)
    imgs_feature_positive = encoder((imgs_positive+1) / 2).view(N, -1)
    imgs_shifted_feature_1 = encoder((imgs_shifted_1+1) / 2).view(batch_size, -1)
    imgs_shifted_feature_2 = encoder((imgs_shifted_2+1) / 2).view(N, -1)

    q = torch.abs(imgs_shifted_feature_1 - imgs_feature)
    k = torch.abs(imgs_shifted_feature_2 - imgs_feature_positive)

    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)

    # then generate queue
    noise_negative = torch.randn(K, generator_latent_dim).cuda()
    imgs_negative, z_negative = generate(generator, noise_negative, return_latent = True)
    imgs_feature_negative = encoder((imgs_negative+1) / 2).view(K, -1)

    shift_negative = make_negative_shift(target_indice, K, total_dim, used_dim)
    shift_negative = navigator(shift_negative)
    imgs_shifted_negative = generate(generator, z_negative + shift_negative)
    imgs_shifted_feature_negative = encoder((imgs_shifted_negative+1) / 2).view(K, -1)

    queue = torch.abs(imgs_shifted_feature_negative - imgs_feature_negative).permute(1,0)
    queue = nn.functional.normalize(queue, dim=0)

    l_pos = torch.einsum('nc,ck->nk', [q, k.permute(1,0)])
    l_neg = torch.einsum('nc,ck->nk', [q, queue.detach()])

    logits = torch.cat([l_pos, l_neg], dim=1)

    # perform hard nagative flipping
    if args.flipping:
        labels =torch.zeros_like(logits).cuda()
        labels[logits > args.thresh] = logits[logits > args.thresh].detach() 
        labels[:,range(N)] = 1

    logits_loss = cross_entropy(logits, labels.float())

    # perform entropy_loss
    if args.entropy:
        entropy_loss = entropy(q)
        loss = logits_loss + entropy_loss
    else:
        entropy_loss = torch.zeros(1) # set loss to zero for logging
        
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # update statistics trackers
    avg_loss.add(loss.item())
    avg_logits_loss.add(logits_loss.item())
    avg_entropy_loss.add(entropy_loss.item())
    
    if global_iter % 100 == 0:
        pbar.write('[{}] avg_loss:{:.3f} logits_loss:{:.3f} entropy_loss:{:.3f}'.format(global_iter, \
        avg_loss.mean(), avg_logits_loss.mean(), avg_entropy_loss.mean()))
        
    if global_iter % 100 == 0:
        # first visualize
        visualize_GAN(generator, navigator, os.path.join(encoder.model_name, "viz", "%06d_W.jpg" % global_iter), used_dim, total_dim)
        torch.save(encoder.state_dict(), os.path.join(encoder.model_name, "encoder.pth"))
        torch.save(navigator.state_dict(), os.path.join(encoder.model_name, "navigator.pth"))
        
    if global_iter >= max_iter:
        out = True
        break