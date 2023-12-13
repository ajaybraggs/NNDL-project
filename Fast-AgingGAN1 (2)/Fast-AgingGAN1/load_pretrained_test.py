from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer

from gan_module import AgingGAN

with open('configs/aging_gan.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    model = AgingGAN(config)