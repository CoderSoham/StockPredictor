import tensorflow as tf

# Checks for available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
