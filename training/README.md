## Model Training

- The training script is implemented in [train_gips.py](train_gips.py)
- Because we precomputed all the embeddings we could omit embedding the InternVL and Clip models into the pipeline, and
  only implemented the attention module with guiding loss head and the different prediction heads.
- We utilize batched training to speedup the training process and reduce the memory consumption.
- To track the training progress we use Tensorboard. The logs are stored in the `logs` directory.