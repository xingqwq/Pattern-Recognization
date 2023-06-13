import imageio
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

imgs = []
images = os.listdir(args.dir)
images.sort()
print(images)
for i in tqdm(images):
    imgs.append(imageio.v2.imread(args.dir + '/' + i))
imageio.mimsave("./{}.gif".format(args.model), imgs, duration=500)