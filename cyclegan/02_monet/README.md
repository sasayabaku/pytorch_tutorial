# monet2photo | CycleGAN
* CycleGANでの、monet2photoのコピー
* `01_horse2zebra`のデータセットを変更したのみ

**Reference**
* [junyanz/pytorch-CycleGAN-and-pix2pix: Image-to-Image Translation in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Prepare Dataset
Download Datasets  
```bash
./datasets/download_dataset.sh
```

## Training Model
If use CUDA , `--cuda` options are enabled.

```bash
python3 train.py --cuda
```

## Prediction
### script
1枚の画像変換用スクリプト  
結果は、`results/predict`の下に保存

**Example**
```bash
python predict.py ./datasets/monet2photo/testA/00010.jpg
```