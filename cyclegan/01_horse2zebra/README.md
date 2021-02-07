# horse2zebra | CycleGAN
CycleGANでの、horse2zebraのコピー  

**Reference**
* [junyanz/pytorch-CycleGAN-and-pix2pix: Image-to-Image Translation in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [PyTorch (15) CycleGAN (horse2zebra) - 人工知能に関する断創録](https://aidiary.hatenablog.com/entry/20180324/1521896184)

## Prepare Dataset
Download Datasets  
```bash
./datasets/download_dataset.sh
```

## Prediction
### script
1枚の画像変換用スクリプト  
結果は、`results/predict`の下に保存

**Example**
```bash
python predict.py ./datasets/horse2zebra/testA/n02381460_1000.jpg
```