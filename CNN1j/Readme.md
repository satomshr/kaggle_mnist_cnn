# CNN1j/Readme.md

- Summary
  - ImageDataGenerator で毎回画像を変形させずに，事前に変形させた画像で学習する

- Files

- Results
  - submit はしていない

- Discussions
  - 初めて val_loss を見たが，val_loss が最小になるのは epochs=5 (val_loss=0.1515)
  - 現時点で最高記録の CNN1g では epochs=30 で val_loss=0.0251 。epoch=19 の val_loss=0.0238 が最小。
  - val_loss の最小を目指すのが正しいなら，今回の取り組みでは不十分。
  - CNN1g をベースに，例えば学習率を小さくすることなどを考えるべきだろうか?
