# CNN1n/Readme.md
In the CNN1l, best conditions (06) are as follows;
- 1st Cond2D ; filter is 7x7
- Use learning rate reducing, starting lr=0.001, factor=0.47
- Use dropout (0.4) after each Cond2D
- Channels are doubled in each Cond2D

Here, based on the condition of CNN1l/06, try various method.

# Training conditions and Result of score
### Common conditions
- Batch size ; 32
- Dropout after Cond2D ; Yes (0.4)
- BatchNormalization after Cond2D ; No

### Training conditions
| No| Conditions | Min of val_loss | Max of val_accuracy | Score |
|:-:| :-- | :-: | :-: | :-: |
|Ref| CNN1l/06 | 0.02138 (epochs=65)| 0.99512 (epochs=68) | 0.99507 (epochs=62)|
| 00| factor = 0.631 |0.02099 (epochs=61) |0.99595 (epochs=65) | |


## Detail
### 00 ; learning rate reducing factor ; 0.47 -> 0.631
Factor of learning rate reducing is changed from 0.47 to 0.631


## Results

## Graphs
### Reference (CNN1l/06)
![graphs of accuracy and loss](../CNN1l/06/CNN1l_06.svg)

### 00 ; Standard condition
![graphs of accuracy and loss](./00/CNN1n_00.svg)
