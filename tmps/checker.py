import tensorflow as tf

print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
print("Devices:", tf.config.list_physical_devices())