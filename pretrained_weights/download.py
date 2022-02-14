import os
import tarfile
import argparse
import shutil

SOURCES = {
    'mnist': 'https://www.dropbox.com/s/rzurpt5gzb14a1q/pretrained_mnist.tar',
    'anime': 'https://www.dropbox.com/s/9aveavgbluvjeu6/pretrained_anime.tar',
    # 'biggan': 'https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar',
}


def download(source, destination):
    tmp_tar = os.path.join(destination, '.tmp.tar')
    # urllib has troubles with dropbox
    os.system(f'wget {source} -O {tmp_tar}')
    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(destination)

    os.remove(tmp_tar)


def main():
    parser = argparse.ArgumentParser(description='Pretrained models loader')
    parser.add_argument('--models', nargs='+', type=str,
                        choices=list(SOURCES.keys()) + ['all'], default=['all'])
    parser.add_argument('--out', type=str, help='root out dir', default = "./pretrained_weights/")

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    models = args.models
    if 'all' in models:
        models = list(SOURCES.keys())

    for model in set(models):
        source = SOURCES[model]
        print(f'downloading {model}\nfrom {source}')
        download(source, args.out)
        
    # then do post process
    shutil.move(os.path.join(args.out, "pretrained", "generators", "SN_Anime"), args.out)
    shutil.move(os.path.join(args.out, "pretrained", "generators", "SN_MNIST"), args.out)

    shutil.rmtree(os.path.join(args.out, "pretrained"))
    
if __name__ == '__main__':
    main()