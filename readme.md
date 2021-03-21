# GIQA: Generated Image Quality Assessment
This is the official pytorch implementation of ECCV2020 "GIQA: Generated Image Quality Assessment" （<https://arxiv.org/abs/2003.08932>）. The major contributors of this repository include Shuyang Gu, Jianmin Bao, Dong Chen, Fang Wen at Microsoft Research Asia.

## Introduction

**GIQA** aims to solve the problem of quality evaluation of a single generated image. In this source, we release the code of our GMM-GIQA and KNN-GIQA which are convenient to use.

## Citation
If you find our code  helpful for your research, please consider citing:
```
@article{gu2020giqa,
  title={GIQA: Generated Image Quality Assessment},
  author={Gu, Shuyang and Bao, Jianmin and Chen, Dong and Wen, Fang},
  journal={arXiv preprint arXiv:2003.08932},
  year={2020}
} 
```

## Getting Started

### Prerequisite
- Linux.
- Pytorch 1.0.0.
- CUDA9.2 or 10.
- sklearn 0.22.2

### Running code
- Download pretrained models [here](https://drive.google.com/drive/folders/17fAzhyQGXwgSJYO1PhmbnSl72FAE4VCJ?usp=sharing).  We provide the LSUN-cat GMM model with PCA95 in this link, if you need more models, please contact me.

- Extract features:

  ```
  python write_act.py path/to/dataset --act_path path/to/activation --pca_rate pca_rate --pca_path path/to/pca --gpu gpu_id
  ```
- Get KNN-GIQA score:

  ```
  python knn_score.py path/to/test-folder --act_path path/to/activation --pca_path path/to/pca --K number/of/nearest-neighbor --output_file output/file/path --gpu gpu_id
  ```

- Get GMM-GIQA score:

  first build the GMM model:

  ```
  python get_gmm.py --act_path path/to/activation --kernel_number number-of-Gaussian-components --gmm_path path/to/gmm
  ```

  then get the GMM-GIQA score:
  ```
  python gmm_score.py path/to/test-folder --gmm_path path/to/gmm --pca_path path/to/pca --output_file output/file/path --gpu gpu_id
  ```

- For all these running bash, if we do not use PCA (such as FFHQ), just remove the pca_rate and pca_path options.

### LGIQA dataset
- The LGIQA dataset contains three sub-dataset, named LGIQA-FFHQ, LGIQA-cat, LGIQA-cityscapes. You can download the cat and cityscapes sub-dataset [here](https://drive.google.com/drive/folders/1NHeFPdswV9lAmLtfiCppExi3oQ5lfSFH?usp=sharing). For security reason, if you need LGIQA-FFHQ dataset, please contact me. 

### test
- To test if you run our code correctly, we provide results of our provided GMM-GIQA model (on LSUN-cat dataset). We put it in the test folder. For test_images, by using the "get GMM-GIQA score command" and our provided model, you can get the results like results.txt. As we pointed out in our paper, the value of the score is meaningless, but you can use the rank of score to compare which has a higher quality. For different GMM models (especially different kernels), the score has a very large range (probably from -10^7 to 10^5), it's normal since we do not directly need the value. And also, for user's dataset, it should notice that the dataset Could Not contain too few images, otherwise, the GIQA score may be inaccurate.

## Reference

[pytorch-fid](https://github.com/mseitzer/pytorch-fid)

