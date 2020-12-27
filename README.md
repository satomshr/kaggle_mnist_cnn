# kaggle_mnist_cnn
Kaggle/MNIST using CNN

- CNN1a
  - Files
    - digit_recognizer_CNN1a.csv
    - digit-recognition_CNN1a.ipynb
    - digit-recognition_CNN1a.py
    - prediction_CNN1a.csv ; results of prediction (prediction and probability)
  - Summary
    - TensorFlow の tutorial (https://www.tensorflow.org/tutorials/images/cnn) をそのまま実装
    - epochs=5
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
