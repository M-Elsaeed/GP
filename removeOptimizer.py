import keras
from tensorflow.python.ops.gen_array_ops import invert_permutation
model = keras.models.load_model("D:/Updated/GP/trainedModels/croppedModelFC.h5")
keras.models.save_model(model, "./croppedModelFCNoOptimizer.h5", overwrite=False, include_optimizer=False)