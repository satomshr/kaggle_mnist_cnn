# kaggle_mnist_cnn
Kaggle/MNIST using CNN

- CNN1a
  - Files
    - digit_recognizer_CNN1a.csv
    - digit-recognition_CNN1a.ipynb
    - digit-recognition_CNN1a.py
    - prediction_CNN1a.csv ; results of prediction (prediction and probability)
    - mismatched_CNN1a.png
  - Summary
    - TensorFlow の tutorial (https://www.tensorflow.org/tutorials/images/cnn) をそのまま実装
    - epochs=5
  - Model Summary
```  
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 576)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                36928     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
```
  - Results
    - 0.98792 (Best!)

- CNN1b
  - Files
    - digit_recognizer_CNN1b.csv
    - digit_recognizer_CNN1b.jpg
    - digit_recognition_CNN1b.ipynb
    - digit_recognition_CNN1b.py
  - Summary
    - CNN1a の epochs を 20 まで上げて，loss や loss_val の値がどう変化するか確認した
    - 一番良かったのは epochs = 6 だった

- CNN1c
  - Files
    - digit_recognizer_CNN1c.csv
    - digit_recognition_CNN1c.ipynb
    - digit_recognition_CNN1c.py
  - Summary
    - CNN1a で epochs=6 にした
  - Results
    - 0.98625 (残念)
    - Saved as Ver.8 on Kaggle

- CNN1e
  - Files
    - digit_recognition_CNN1e.ipynb
    - digit_recognition_CNN1e.py
    - train_CNN1e.txt ; results of train
      - epochs = 5 くらいで saturate している
      - val_accuracy が一番大きいのは epochs = 14
    - train_CNN1e.png ; graph of above
    - digit-recognition_CNN1e.ipynb
    - digit-recognition_CNN1e.py
    - digit_recognition_CNN1e_epochs06.csv ; epochs = 6
    - prediction_CNN1e_epochs06.csv
    - digit_recognition_CNN1e_epochs14.csv ; epochs = 14
    - prediction_CNN1e_epochs14.csv
  - Summary
    - CNN1a で，最初の kernel_size を (3,3) から (5,5) に変更した
    - 理由は，太い数字に対する認識を強化するために，フィルタのサイズを大きくした
  - Results
    - epochs=6 ; 0.98653
    - epochs=14 ; 0.99035 (更新!! 762/2105 = 0.362)
    - Saved as Ver.9 on Kaggle (epochs=14)

- CNN1f
  - See [CNN1f/Readme.md](./CNN1f/Readme.md)
  - CNN1e で，チャンネル数を増やした
  - epochs = 22 ; 0.99050 (Best! 770/2158 = 0.3568)

- CNN1g
  - See [CNN1g/Readme.md](./CNN1g/Readme.md)
  - CNN1f で，ImageDataGenerator を使って画像を変形させた
  - epochs = 30 ; 0.99228 (570 / 2182 = 0.2612)

- CNN1h
  - See [CNN1h/Readme.md](./CNN1h/Readme.md)
  - X_cv を random_transform した
  - ImageDataGenerator の変形範囲を増やした
  - 試しに Dropout を入れてみた
  - スコアは更新できず
  - epochs = 35 ; 0.99157

- CNN1i
  - See [CNN1i/Readme.md](./CNN1i/Readme.md)
  - チャンネル数を増やした
  - スコアは更新できず
  - epochs = 30 ; 0.99185 (not good)

- CNN1j
  - See [CNN1j/Readme.md](./CNN1j/Readme.md)
  - ImageDataGenerator で毎回変形させずに，事前に変形させた画像で学習する

- CNN2a
  - Files
  - Summary
    - LeNet-5 をトライ
    - http://datahacker.rs/lenet-5-implementation-tensorflow-2-0/
    - https://ohke.hateblo.jp/entry/2019/03/16/143000

- misc
  - check_prediction1.py ; chech prediction and see images
  - show_history.py ; create graph of history (accracy, val_accuracy, loss, val_loss)
