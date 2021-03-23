set -ex
CUDA_VISIBLE_DEVICES="2" python train.py --name cdegan \
       --dataset_mode torchvision --batch_size 32 --eval_size 256 --dataroot None \
       --model cdegan \
       --gpu_ids 0 \
       --download_root ./datasets/cifar10 --dataset_name CIFAR10 \
       --crop_size 32 --load_size 32 \
       --d_loss_mode vanilla lsgan --g_loss_mode nsgan vanilla lsgan --which_D S \
       --lambda_f 0.1 --lambda_weight 1\
       --d_candi_num 2 --g_candi_num 1 \
       --z_type Uniform --z_dim 100 \
       --netD DCGAN_cifar10 --netG DCGAN_cifar10 --ngf 128 --ndf 128 --g_norm none --d_norm batch \
       --init_type normal --init_gain 0.02 \
       --no_dropout \
       --D_iters 3 \
       --D_different True --G_different True\
       --print_freq 2000 --display_freq 2000 --score_freq 5000 --display_id -1 --save_giters_freq 100000\
       --score_name IS --evaluation_size 50000 --fid_batch_size 500 \
#       --use_pytorch_scores
#