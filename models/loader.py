###########################################################
# The file is for load the generative model you need
###########################################################
import torch

def get_encoders(nc, args):
    if args.dataset == "MNIST": # resolution 32 * 32
        from models.DisCo.encoder import Baseline_Encoder_1
        encoder = Baseline_Encoder_1(nc = nc, z_dim = args.z_dim)
    else:
        from models.DisCo.encoder import Baseline_Encoder
        encoder = Baseline_Encoder(nc = nc, z_dim = args.z_dim)
    
    return encoder
    
def get_generator(args):
    # first judge the generative model type
    if args.G == "stylegan":
        generator = get_style_generator(args)
        latent_dim = 512
        nc = 3 # number of channel
        
    elif args.G == "sngan":
        from models.SNGAN.load import load_model_from_state_dict
        
        latent_dim = 128
        
        if args.dataset == "MNIST":
            nc = 1 # number of channel
            generator = load_model_from_state_dict("../pretrained_weights/SN_MNIST")
            generator = generator.model
            generator.size = 32

        elif args.dataset == "Anime":
            nc = 3 # number of channel
            generator = load_model_from_state_dict("../pretrained_weights/SN_Anime")
            generator = generator.model
            generator.size = 64
        else:
            raise NotImplementedError
    
    elif args.G == "biggan":
        raise NotImplementedError
        
    elif args.G == "vae":
        raise NotImplementedError
        
    elif args.G == "glow":
        raise NotImplementedError
        
    return generator, latent_dim, nc

def get_style_generator(args):
    # first we define generator
    from models.StyleGAN2.models import Generator
    dataset = args.dataset
    
    if dataset == "isaac":
        config = {"latent" : 512, "n_mlp" : 3, "channel_multiplier": 2}
        generator = Generator(
                size= 128,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                small = False,
                channel_multiplier=config["channel_multiplier"]
            )
    elif dataset == "ffhq":
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        generator = Generator(
                size= 1024,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
    elif dataset == "ffhq_low":
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 1}
        generator = Generator(
                size= 256,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
    elif dataset in ["cat", "church"]:
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        generator = Generator(
                size= 256,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
    elif dataset == "car":
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        generator = Generator(
                size= 512,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
    else:
        config = {"latent" : 512, "n_mlp" : 3, "channel_multiplier": 4}
        generator = Generator(
                size= 64,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                small = True,
                channel_multiplier=config["channel_multiplier"]
            )

    if dataset == "shapes3d":
        generator.load_state_dict(torch.load("./pretrained_weights/shapes3d/%01d.pt" % args.index))
    elif dataset == "noisy":
        generator.load_state_dict(torch.load("./pretrained_weights/noisy/%01d.pt" % args.index)['g_ema'])
    elif dataset == "mpi3d":
        generator.load_state_dict(torch.load("./pretrained_weights/mpi3d/%01d.pt" % args.index))
    elif dataset == "cars3d":
        generator.load_state_dict(torch.load("./pretrained_weights/cars/%01d.pt" % args.index))
    elif dataset == "ffhq":
        generator.load_state_dict(torch.load("./pretrained_weights/ffhq/generator.pt")['g_ema'])
    elif dataset == "ffhq_low":
        generator.load_state_dict(torch.load("./pretrained_weights/convert_ffhq_256_config-e.pt")["g_ema"])
    elif dataset == "cat":
        generator.load_state_dict(torch.load("./pretrained_weights/convert_cat_256_config-f.pt")["g_ema"])
    elif dataset == "church":
        generator.load_state_dict(torch.load("./pretrained_weights/convert_church_256_config-f.pt")["g_ema"])
    elif dataset == "car":
        generator.load_state_dict(torch.load("./pretrained_weights/convert_car_512_config-f.pt")["g_ema"])
    else:
        raise NotImplementedError
        
    return generator