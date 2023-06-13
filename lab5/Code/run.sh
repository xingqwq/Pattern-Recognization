python main.py --model GAN --epoch 2000 --optimizer sgd
python main.py --model GAN --epoch 1000 --optimizer adam
python main.py --model GAN --epoch 1000 --optimizer rmrsp
python main.py --model WGAN --epoch 1000 --optimizer rmrsp
python main.py --model WGAN-GP --epoch 1000 --optimizer rmrsp

python gifGenerate.py --dir ./png/GAN_adam --model GAN_adam
python gifGenerate.py --dir ./png/GAN_sgd --model GAN_sgd
python gifGenerate.py --dir ./png/GAN_rmrsp --model GAN_rmrsp
python gifGenerate.py --dir ./png/WGAN_rmrsp --model WGAN_rmrsp
python gifGenerate.py --dir ./png/WGAN-GP_rmrsp --model WGAN-GP_rmrsp