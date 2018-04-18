import os

source_dir = "./train"
train_dir = "./data/train"
valid_dir = "./data/validation"

os.makedirs("%s/dogs" % train_dir)
os.makedirs("%s/cats" % train_dir)
os.makedirs("%s/dogs" % valid_dir)
os.makedirs("%s/cats" % valid_dir)

for i in range(10000):
    os.rename("%s/dog.%d.jpg" % (source_dir, i + 1),
              "%s/dogs/dog%04d.jpg" % (train_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, i + 1),
              "%s/cats/cat%04d.jpg" % (train_dir, i + 1))

for i in range(2500):
    os.rename("%s/dog.%d.jpg" % (source_dir, 10000 + i + 1),
              "%s/dogs/dog%04d.jpg" % (valid_dir, i + 1))
    os.rename("%s/cat.%d.jpg" % (source_dir, 10000 + i + 1),
              "%s/cats/cat%04d.jpg" % (valid_dir, i + 1))
