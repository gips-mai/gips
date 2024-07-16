## Model Implementation

- The module combining all steps of the model pipeline is called Gips and is implemented in [gips.py](gips.py)
- Because we precomputed all the embeddings we could omit embedding the InternVL and Clip models into the pipeline, and
  only implemented the attention module with guiding loss head and the different prediction heads.

### Attention Module

- The attention module is implemented in [attention_module.py](modules%2Fattention_module.py)
- For additionally guiding the attention module during training, we followed the approach of the G^3 paper and
  implemented an additional guiding head, which guides the attention module to select the correct clues for a given
  image. This is implemented in [guiding_head.py](modules%2Fheads%2Fguiding_head.py).

### Prediction Heads

- The different headas are implemented in [heads](modules%2Fheads). We implemented two approaches:
  1. From OSV5M, we adopted the hybrid head which combines classification and regression to predict the latitude and
     longitude of an image. This is implemented in [geoloc_head.py](modules%2Fheads%2Fgeoloc_head.py).
  2. Because the hybrid head has caused us problems during training, we also implemented a simpler regression head which
     directly regresses the latitude and longitude of an image. This is implemented in [easy_reg_head.py](modules%2Fheads%2Feasy_reg_head.py).

