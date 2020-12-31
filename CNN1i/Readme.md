# CNN1i/Readme.md

- Summary
  - CNN1h をベースにして，Dropout を削除
  - 代わりに，チャンネル数を増やした

- Files ; saved as ver.12 on kaggle

- Results
  - epochs = 24 ; 0.99025
  - epochs = 30 ; 0.99185 (not good)
  - epochs = 85 ; 0.99178

- Discussions
  - ImageDataGenerator で epoch 毎に新しく変形させた画像を作っているが，そのために目指すべきパラメタに収束していないのではないか?
  - あらゆる画像に対して適切な予測をするパラメタを作るという意味では，毎回変形させた画像で学習するというのも意義があるとは思うが…
  - accuracy のグラフを見ると，まだ微増している。過学習になっていないというメリットはあるが，収束が遅い気がする
  - 次のアクションとして，変形した画像を固定にして，それで学習させてみてはどうだろうか?
