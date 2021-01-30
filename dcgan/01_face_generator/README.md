# Face Generator | DCGAN

PyTorch 公式 チュートリアルの 「DCGAN Tutorial」 のコピー 

**Reference**
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# Contents
## main.py
### Generator(class)
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
