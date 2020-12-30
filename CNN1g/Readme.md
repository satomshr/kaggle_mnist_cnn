# CNN1g/Readme.md

- Summary
  - モデルのパラメータは, CNN1f と同じ
  - ImageDataGenerator で画像の変形を行い, model.fit_generator() で変形した画像で学習させる (一見, イジメだね)
  - val_accuracy を更新したときに weight を保存し, その weight を用いて一括で predict するようなスクリプトにした

- Files ; Saved as Version 11 on Kaggle

- Results
  - epochs = 50 ; 0.99196
  - epochs = 39 ; 0.99067
  - epochs = 30 ; 0.99228 (570 / 2182 = 0.2612)
  - epochs = 24 ; 0.99128
  - epochs = 18 ; 0.99139
