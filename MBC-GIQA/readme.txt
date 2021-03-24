Important: I didn't plan to release the code of MBC-GIQA since the performance is worse than GMM-GIQA and KNN-GIQA. Someone asks for it, so I release the training and test source code of MBC-GIQA, it needs to prepare data from the GAN training procedure. Please refer to our paper for more details. However, for the time limitation, I havn't cleaned the MBC-GIQA code yet.


training command:
python gen_image_mulcla.py -a vgg19_bn_mulcla --epochs 112 --schedule 28 56 84 --gamma 0.1 --checkpoint checkpoints/stylegan/m2_192_8bin/ --gpu-id 0,1,2,3 --train_batch 512 --lr 0.05 --workers 64 --train_size 192 --add_gt 1 --num_class 8

test command:
python -m pdb gen_image_mulcla_test.py -a vgg19_bn_mulcla --evaluate --resume checkpoints/stylegan/vgg19_bn_gt_mulcla1/model_best.pth.tar --test_file /mnt/blob/datasets/generation_results/cat/all_cat_test_image.txt --test_batch 100 --root_path /mnt/blob/datasets/generation_results/cat/all_cat_test_image/ --gpu-id 0 --train_size 192 --num_class 8
