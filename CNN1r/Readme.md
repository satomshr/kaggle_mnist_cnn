# CNN1r/Readme.md

## Summary / 要約
1. Train and predict as usual.
2. Re-train the model using train data of low predict probability, and re-predict the label of test data of low probability


1. 普通に学習と予測を行う
2. 予測の確率が低い訓練データを使って再学習し, 確率の低いテストデータの再予測をする

## Preparation / 準備
### Probabilities of right answers and wrong answers / 正しい答えと間違った答えの確率
![正しい答えと間違った答えの確率](./graph_proba1.svg)

![正しい答えと間違った答えの確率(拡大)](./graph_proba2.svg)

- 青い線 ; 正しい答えに対する確率
- 赤い線 ; 予測された確率 (`tf.keras.Sequential.predict_proba()`)のうち最大のもの

グラフは, 正しい答えに対する確率 (青い線の値) で逆順にソートしている. 確率が 0.5 を下回ると, 誤答が増えてくる. 自信を持って間違えている (赤い線の値が高い) 場合と, 迷っている (青い線と赤い線が同じくらいの値) 場合があることが分かる.

訓練データは答えが分かっているので, 正しい答えに対する確率が低い画像を識別できる. しかしテストデータは (当たり前だが) 正答が分からないので, テストデータから間違っていそうなデータだけ抽出して再予測することはできない.

対策として考えられるのは,

- 最大の確率が例えば 0.8 以下の全ての画像に対して, 再予測する
- 全部のテストデータに対して再予測し, 1 回目の予測結果とアンサンブル学習をする

### 確率の低い画像データ
![確率の低い画像データ](./ans_max_digit_image.svg)

正しい答えに対する確率が低いほうから, 150 個のデータを示す。左上が確率が高いほう, 右下が確率が低いほうで, 画像の上の数字は, 左側が正しい答え, 右側が確率が最大だった答え.

これらの画像データで学習すれば, これらの画像データに対する正答率は向上する. しかし, もともと確率が高かった正答の画像を, 誤った結果にしてしまう危険性もある.

### 確率の低いデータのラベルの分布
![確率の低いデータのラベルの分布](./label_low_probability.svg)

正答の確率の低い画像データ (低いほうから 150 個) で, ラベルごとの個数をプロット. x 軸は正答のラベル, y 軸は個数. 1, 4, 7, 9 のデータで, 誤りが多い.

### まとめ
- 予測を間違ったテストデータのみを抽出することはできないので, 再予測する場合は
  - 最大の確率が例えば 0.8 以下のすべての画像に対して再予測する
    - この場合，自信を持って間違えているデータは, 間違えたまま (あきらめ)
  - 全部のテストデータに対して再予測し, 1 回目の予測結果とアンサンブル学習をする
    - 例えば最初の予測で 3 回分, 再予測で 2 回分, 合計 5 回分のデータでアンサンブルするなど, 最初の予測の重みを重めにするなど, 工夫が必要か

## 再予測の条件の検討
### 転移学習の条件と, 損失関数

#### Freeze = none
![Freeze=none](./freeze_none.png)

#### Freeze = layer1
![Freeze=layer1](./freeze_L1.png)

#### Freeze = layer1, layer2
![Freeze=layer1, layer2](./freeze_L1_L2.png)

#### Freeze = layer1, layer2, layer3
![Freeze=layer1, layer2, layer3](./freeze_L1_L2_L3.png)

#### まとめ
- Freeze = none ; accuracy, loss が改善しすぎていて, 過学習になっている
- Freeze = layer1 ; ちょうどいい感じ
- Freeze = layer1, layer2 (, layer3) ; accuracy が不足. 複雑さを表現しきれていない

以上から Freeze = layer1 とする.

### epoch 数
![Freeze=layer1](./freeze_L1_150.png)

Freeze = layer1 とし, epochs=150 で計算. epochs=100 くらいで改善が止まっている感じ.

### まとめ
転移学習は, Freeze = layer1 とし, epochs = 100 とする.

## 00
### Summary
- 通常の学習と, 確率の低い画像データを使って転移学習 (条件は上記)
- 通常の学習と転移学習を各 5 回ずつ行い, 各結果を 0 ～ 5 個ずつ使ってアンサンブル学習
- This script is saved as Ver.19 on kaggle.

### train_data の予測結果
![Results of ensamble training](./00/ensamble_results_soft.svg)
![Results of ensamble training](./00/ensamble_results_hard.svg)

1st training を 5 回, transfer training を 3 回使って ensamble training (soft) が一番良かった. 次は 1st training を 5 回, transfer training を 4 回使って ensamble training (soft).

### Results
|1st training|Transfer training|Soft/Hard|Score|No|
|:-:|:-:|:-:|:-:|:-:|
| 5 | 3 | Soft | 0.99571 | 239 / 2950 |
| 5 | 4 | Soft | 0.99578 | 231 / 2950 (=0.0783)|
| 5 | 5 | Soft | 0.99575 | |
| 5 | 0 | Soft | 0.99528 | |

### Transfer training の成果は出ているのか?
スコアが上がったのは, 単に ensamble するデータが増えただけではないか? の検証 (データは全て soft ensamble)

|Ensamble No|1st training|Transfer training|Score|
|:-:|:-:|:-:|:-:|
| 6 | 5 | 1 | 0.9977619047619047 |
| 6 | 4 | 2 | **0.9978095238095238** |
| 6 | 3 | 3 | 0.9977857142857143 |
| 6 | 2 | 4 | 0.9958571428571429 |

微妙だが, データセット数 = 6 のとき, 1st training を 4 回, Transfer training を 2 回のときが, 一番スコアが良い.

下記に相関係数を示すが, 1st training どうしは 0.995, 1st training と Transfer training は 0.954, Transfer training どうしは 0.963 となっており, わずかではあるが相関係数は低いので, ensamble の効果はあるはず.

```
5-1
[[1.         0.99511309 0.9945586  0.99540225 0.99538242 0.95392299]
 [0.99511309 1.         0.99518185 0.99484475 0.99467612 0.95462115]
 [0.9945586  0.99518185 1.         0.99513058 0.99418213 0.95508154]
 [0.99540225 0.99484475 0.99513058 1.         0.99517363 0.95427125]
 [0.99538242 0.99467612 0.99418213 0.99517363 1.         0.95446136]
 [0.95392299 0.95462115 0.95508154 0.95427125 0.95446136 1.        ]]

 4-2
 [[1.         0.99511309 0.9945586  0.99540225 0.95392299 0.9526449 ]
  [0.99511309 1.         0.99518185 0.99484475 0.95462115 0.95334551]
  [0.9945586  0.99518185 1.         0.99513058 0.95508154 0.95259153]
  [0.99540225 0.99484475 0.99513058 1.         0.95427125 0.95251388]
  [0.95392299 0.95462115 0.95508154 0.95427125 1.         0.9634931 ]
  [0.9526449  0.95334551 0.95259153 0.95251388 0.9634931  1.        ]]
```  

## 01
### Summary
`ImageDataGenerator` のパラメータをランダムサーチで最適化する.

### 計算
- 1 epoch = 20s
- 1 条件の計算は 70 epochs = 1400s (23.34min) (本番の計算は 65 epochs にした)

### Files
- history_ImageDataGenerator.csv ; accuracy and loss of optimal condition
- results_ImageDataGenerator.csv ; parameters of ImageDataGenerator and val_loss
  - val_loss の大きい条件と小さい条件は存在するようだ

### 9 回繰り返した中での最適の accuracy と loss
![accuracy and loss](./01/results_ImageDataGenerator_opt.png)

## 02
### Summary
- `GPyOpt` を使って，`ImageDataGenerator` のパラメータの最適値を探す

### Files
- myBopt.csv
  - 1 回目の最適化. `horizontal_flip` と `vertical_flip` も変数にした.
  - 結局のところ, `horizontal_flip = False`, `vertical_flip = False` のほうが良かった
- myBopt2.csv
  - 2 回目の最適化. `horizontal_flip = False`, `vertical_flip = False` として合計 20 回計算
  - 結果として, パラメータの振り幅を小さくするのが良いという, 「そう言われればそうだよね」という結果が出た。ただそれだと, 過学習か否かの判断がつかない.
  - 過学習を防ぎつつ, `val_loss` を防ぐには, ランダムサーチにして, 損失関数を見ながら, 最適の値を探しに行くのが良いか?

## 03
### Summary
- ランダムサーチで `ImageDataGenerator` のパラメータの最適化をする
- 00 と同様に, 通常の学習と, 確率の低い画像データを使って転移学習 (条件は上記)
- 通常の学習と転移学習を各 5 回ずつ行い, 各結果を 0 ～ 5 個ずつ使ってアンサンブル学習
- (This script is saved as Ver.19 on kaggle.)

### ImageDataGenerator のランダムサーチ
history 以下のファイルは, `ImageDataGenerator` のランダムサーチの結果. `val_loss` が一番小さかったのは, 以下の組み合わせ (カッコ内は従来の数値)

- rotation_range = 11 (30)
- width_shift_range = 0.13 (0.20)
- height_shift_range = 0.25 (0.20)
- shear_range = 0.37 (0.20)
- zoom_range = 0.17 (0.20)

### train_data の予測結果
![Results of ensamble training](./03/ensamble_results_soft.svg)
![Results of ensamble training](./03/ensamble_results_hard.svg)

00 と違って, 転移学習の結果を入れすぎると, 精度が下がる. 通常学習を 5, 転移学習を 0, 1, 2 が一番良い.
soft と hard の違いはあまりない.

### Results
|1st training|Transfer training|Soft/Hard|Score|No|
|:-:|:-:|:-:|:-:|:-:|
| 5 | 0 | Soft | 0.99578 |  |
| 5 | 0 | Hard | 0.99607 | 166 / 2223 (=0.0746) |
| 5 | 1 | Soft | 0.99582 |  |
| 5 | 1 | Hard | 0.99592 |  |
| 5 | 2 | Soft | 0.99603 |  |
| 5 | 2 | Hard |  |  |

## テスト
### フォルダ
- [test](./test/)

### ファイル
- CNN1r_a.svg, CNN1r_b.svg ; 2 回目の学習の結果

![./test/CNN1r_a.svg](./test/CNN1r_a.svg)

![./test/CNN1r_b.svg](./test/CNN1r_b.svg)

学習結果を見ると, `loss` の減少は続いているが `val_loss` の減少は途中で止まっている. 過学習になっている可能性がある. オンラインでグラフを見ながらパラメータを調整する必要がある.

## time
時間関連の関数のテスト. ある一定時間で `while` 抜ける実験

### フォルダ
- [time](./time/)

## gpyopt
GPyOpt のテスト.

### フォルダ
- [gpyopt](./gpyopt/)
