# CNN1h/Readme.md

- Summary
  - ImageDataGenerator の変形範囲を増やした
    - rotation_range : 30 -> 35
    - width_shift_range : 0.2 -> 0.25
    - height_shift_range : 0.2 -> 0.2 (高さ方向にずらし過ぎると，数字の形が大きく変わる気がした)
  - X_cv を random_transform した
  - 試しに Dropout を入れた (Conv2D の後ろに，0.2)

- FYI
  - old 以下のフォルダは, epochs=100 で計算した結果の一部。予測結果をダウンロードできなかった。ただ，結果的には epochs=50 で十分だったので，再計算した。

- Results
  - epochs = 35 ; 0.99157
  - epochs = 29 ; 0.99071

- Discussions
  - accuracy や val_accuracy のグラフを見ると，まだまだ精度は上昇中のような気がする
