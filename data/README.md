# Data

The scripts automatically download **CIFAR-10** from the official torchvision mirror into `data/raw/`. The downloaded files are intentionally ignored by Git.

CIFAR-10 contains 60,000 RGB images (32×32 pixels) in 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The canonical 50,000-image training set is deterministically split into 45,000 training and 5,000 validation images (seed 42); the canonical 10,000-image test set remains untouched.

Source: https://www.cs.toronto.edu/~kriz/cifar.html
