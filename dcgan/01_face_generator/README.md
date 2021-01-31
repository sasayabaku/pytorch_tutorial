# Face Generator | DCGAN

PyTorch 公式 チュートリアルの 「DCGAN Tutorial」 のコピー 

**Dataset**  
https://www.kaggle.com/jessicali9530/celeba-dataset

**Reference**
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## Demo

### Prepare Dataset
1.  [This Page or kaggle API](https://www.kaggle.com/jessicali9530/celeba-dataset) を使って、データセットのダウンロード
1. `01_face_generator/dataset`を作成し、ダウンロードしたデータを解凍

### Training
#### script
```bash
python main.py 
```

### Visualization
<a href="" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/jupyter-Notebook-important?logo=jupyter" />
</a>

## Contents
### main.py
#### Generator(class)
GANのGeneratorモデルクラス

### Discriminator(class)
GANのDiscriminatorモデルクラス

### argparser(function)
引数処理

### main(function)
* Generator / Discriminatorの訓練タスク 
* 500iterごとに、Generatorが生成する画像を保存
* 訓練後に、
  * 各iterでのLoss値を、.h5ファイルで保存
  * 各iterでの生成画像 / Generator / Discriminatorを.pthファイル保存
