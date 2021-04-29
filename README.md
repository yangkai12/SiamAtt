# SiamAtt
This is an official implemention for “SiamAtt: Siamese attention network for visual tracking”.
![image](![SiamAtt](https://user-images.githubusercontent.com/25238475/116553658-f0e31080-a92c-11eb-854e-a9d16ad6e4bd.png)
)
## Dependencies
* Python 3.7
* PyTorch 1.0.0
* numpy
* CUDA 10
* skimage
* matplotlib
## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/) ([BaiduYun](https://pan.baidu.com/s/1nXe6cKMHwk_zhEyIm2Ozpg), extract code: h964.)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)

#### Training:
```bash
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

#### Testing:
```
python ../tools/test.py 
```


References

[1]SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks. Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

## Acknowledgment
Our anchor-free tracker is based on [PySot](https://github.com/STVIR/pysot). We sincerely thank the authors Bo Li for providing these great works.

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{Siamatt,
	author = {Kai Yang, Zhenyu He, Zikun Zhou, Nana Fan},
	title = {SiamAtt: Siamese attention network for visual tracking},
	booktitle = {Knowledge-based system},
	month = {June},
	year = {2020}
	}
