from experiments.training.train_batch import train_batch

for batch_size in [512]:
    train_batch(3e-3, batch_size)
