import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

def generate(generator, rep, return_latent = False):
    if generator.type == "stylegan":
        if generator.Z:
            imgs, _ = generator([rep])
            latent = rep
        if generator.W:
            if return_latent:
                latent = generator.style(rep)
            else:
                latent = rep
            imgs, _ = generator(styles = [latent], input_is_latent=True)
        if generator.S:
            raise NotImplementedError
            
    elif generator.type == "sngan":
        imgs = generator(rep)
        latent = rep
        
    else:
        raise NotImplementedError
    
    if return_latent:
        return imgs, latent
    else:
        return imgs

def visualize_GAN(G, navigator, path, used_dim = 64, total_dim = 64):
    plt.rcParams['figure.dpi'] = 1000
    navigator.eval()
    G.eval()
    
    with torch.no_grad():
        noise = torch.randn(1,G.generator_latent_dim).cuda()
        imgs, style = generate(G, noise, return_latent = True)

        samples = []
        # only visualize the used directions
        for k in range(used_dim):
            interpolation = torch.arange(-16, 16, 3)
            for val in interpolation:
                z = torch.zeros(total_dim).cuda()
                z[k] = val
                shift = navigator(z)
                sample = generate(G, style + shift)
                sample = ((sample+1) / 2).clamp(0,1).cpu()
                samples.append(sample)

        samples = torch.cat(samples, dim = 0)
        output = make_grid(samples, nrow= 11, padding = 0)

        for i, y in enumerate(range(10, 10 + used_dim * G.size, G.size)):
            plt.text(1, y, str(i), color="red", fontsize=1)

        out = output.detach().permute(1,2,0).numpy()
        plt.axis('off')
        plt.imshow(out)
        plt.savefig(path)
        del output
        
# def visualize_Anime(G, mlp, path, line_number = 64, VAE_dim = 64):
#     plt.rcParams['figure.dpi'] = 2000
#     mlp.eval()
#     G.eval()
#     G.size = 64
    
#     with torch.no_grad():
#         style = torch.randn(1, 128).cuda()

#         # first do W
#         samples = []
#         for k in range(line_number):
#             interpolation = torch.arange(-16, 16, 3)
#             for val in interpolation:
#                 z = torch.zeros(VAE_dim).cuda()
#                 z[k] = val
#                 shift = mlp(z)
#                 sample = G(style + shift)
#                 sample = ((sample+1) / 2).clamp(0,1).cpu()
#                 samples.append(sample)

#         samples = torch.cat(samples, dim = 0)
#         output = make_grid(samples, nrow= 11, padding = 0)

#         for i, y in enumerate(range(10, 10 + line_number * G.size, G.size)):
#             plt.text(1, y, str(i), color="red", fontsize=1)

#         out = output.detach().permute(1,2,0).numpy()
#         plt.imshow(out)
#         plt.savefig(path)
#         del output
        
# def visualize_onehot_glow(G, mlp, path, visual_noise, line_number = 64, VAE_dim = 64):
#     plt.rcParams['figure.dpi'] = 1000
#     mlp.eval()
#     G.eval()
    
#     with torch.no_grad():
#         visual_noise[-1] = torch.randn(1, 96, 4, 4).cuda() * 0.7
        
#         samples = []
#         for k in range(line_number):
#             visual_noise_temp = visual_noise.copy()
#             interpolation = torch.arange(-16, 16, 3)
#             for val in interpolation:
#                 z = torch.zeros(VAE_dim).cuda()
#                 z[k] = val
#                 shift = mlp(z).view(1, 96, 4, 4)
#                 visual_noise_temp[-1] = visual_noise[-1] + shift
#                 sample = G.reverse(visual_noise_temp)
#                 sample = ((sample - torch.min(sample))/ (torch.max(sample) - torch.min(sample))).cpu()
#                 samples.append(sample)

#         samples = torch.cat(samples, dim = 0)
#         output = make_grid(samples, nrow= 11, padding = 0)

#         for i, y in enumerate(range(10, 10 + line_number * 64, 64)):
#             plt.text(1, y, str(i), color="red", fontsize=1)

#         out = output.detach().permute(1,2,0).numpy()
#         plt.imshow(out)
#         plt.savefig(path)
#         del output
        
# def visualize_onehot_VAE(G, mlp, path, visual_noise, line_number = 64, VAE_dim = 64):
#     plt.rcParams['figure.dpi'] = 1000
#     mlp.eval()
#     G.eval()
    
#     with torch.no_grad():
#         samples = []
#         for k in range(line_number):
#             visual_noise_temp = visual_noise.unsqueeze(0)
#             interpolation = torch.arange(-16, 16, 3)
#             for val in interpolation:
#                 z = torch.zeros(VAE_dim).cuda()
#                 z[k] = val
#                 shift = mlp(z)
#                 sample = G(visual_noise_temp + shift)
#                 sample = torch.sigmoid(sample).cpu()
#                 samples.append(sample)

#         samples = torch.cat(samples, dim = 0)
#         output = make_grid(samples, nrow= 11, padding = 0)

#         for i, y in enumerate(range(10, 10 + line_number * 64, 64)):
#             plt.text(1, y, str(i), color="red", fontsize=1)

#         out = output.detach().permute(1,2,0).numpy()
#         plt.imshow(out)
#         plt.savefig(path)
#         del output