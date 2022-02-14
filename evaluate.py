import numpy as np
import os
import gin
import torch
import json
import argparse

from data.ground_truth import named_data
from models.loader import get_encoders

def write_text(metric, result_dict, print_txt, file):
    file = file.replace('model','eval')
    if os.path.exists(file):
        with open(file,'r') as f:
            new_dict = json.load(f)
    else:
        new_dict = {}
    new_dict[metric] = result_dict
    if print_txt:
        with open(file,'w') as f:
            json.dump(new_dict,f)

def evaluate(net,
    dataset = None,
    beta_VAE_score = False,
    dci_score = False,
    factor_VAE_score = False,
    MIG = False,
    print_txt = False,
    txt_name = "metric.json"
    ):
    
    def _representation(x):
        x = torch.from_numpy(x).float().cuda()
        x = x.permute(0,3,1,2)
        z = net(x.contiguous()).squeeze()
        return z.detach().cpu().numpy()

    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        write_text("beta_VAE_score", result_dict,print_txt, net.model_name + txt_name)
        gin.clear_config()
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
        result_dict = compute_dci(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("dci score:" + str(result_dict))
        write_text("dci_score",result_dict,print_txt,net.model_name + txt_name)
        gin.clear_config()
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate",10000)
            gin.bind_parameter("factor_vae_score.num_train",10000)
            gin.bind_parameter("factor_vae_score.num_eval",5000)
            gin.bind_parameter("factor_vae_score.batch_size",64)
            gin.bind_parameter("prune_dims.threshold",0.05)
        result_dict = compute_factor_vae(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("factor VAE score:" + str(result_dict))
        write_text("factor_VAE_score",result_dict,print_txt,net.model_name + txt_name)
        gin.clear_config()
    if MIG:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("mig.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_mig(dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MIG score:" + str(result_dict))
        write_text("MIG",result_dict,print_txt,net.model_name + txt_name)
        gin.clear_config()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluation codes")
    parser.add_argument("--dataset", type=int, default=0,
                        help="type of dataset")
    parser.add_argument("--exp_name", type=str, default="train",
                        help="experiment name")
    parser.add_argument("--index", type=int, default=0,
                    help="the index of pretrained generator: range from 0-5")
    parser.add_argument("--z_dim", type=int, default=64,
                        help="the dimension of the output for the encoder")
    args = parser.parse_args()
    
    # get the name of dataset
    assert 0 <= args.dataset or args.dataset <= 4, "only dataset with ground truth can be eval!"
    
    choices = ["shapes3d", "mpi3d", "cars3d", "color", "noisy"]
    dataset = choices[args.dataset]
    args.dataset = dataset

    # get the encoder (Contrastor)
    encoder = get_encoders(3, args)
    
    encoder.load_state_dict(torch.load("./experiments/%s/%s/encoder.pth" % (args.dataset, args.exp_name + str(args.index))))
    encoder.model_name = "./experiments/%s/%s/" % (dataset, args.exp_name + str(args.index))
    encoder.cuda()
    print("model create successfully!")
    
    with gin.unlock_config():
        gin.bind_parameter("dataset.name", dataset)
    dataset_ = named_data.get_named_ground_truth_data()
    evaluate(encoder,
        dataset = dataset_,
        beta_VAE_score = True,
        dci_score = True,
        factor_VAE_score = True,
        MIG = True,
        print_txt = True
        )