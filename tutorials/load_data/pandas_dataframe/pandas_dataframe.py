import numpy as np
import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)

print(df.head())

print(df.dtypes)

target = df.pop('target')

numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
print(numeric_features.head())

print(tf.convert_to_tensor(numeric_features))

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(numeric_features))

print(normalizer(numeric_features.iloc[:3]))

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
  print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

model = get_basic_model()
model.fit(numeric_batches, epochs=15)

numeric_features_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in dict(numeric_features).items()}
target_array = target.to_numpy()[:, tf.newaxis]

numeric_dict_ds = tf.data.Dataset.from_tensor_slices((numeric_features_dict, target_array))

print(len(numeric_features_dict))

for row in numeric_dict_ds.take(3):
  print(row)

class MyModel(tf.keras.Model):
  def __init__(self):
    # Create all the internal layers in init.
    super().__init__()

    self.normalizer = tf.keras.layers.Normalization(axis=-1)

    self.seq = tf.keras.Sequential([
      self.normalizer,
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    self.concat = tf.keras.layers.Concatenate(axis=1)

  def _stack(self, input_dict):
    values = []
    for key, value in sorted(input_dict.items()):
      values.append(value)

    return self.concat(values)

  def adapt(self, inputs):
    # Stack the inputs and `adapt` the normalization layer.
    inputs = self._stack(inputs)
    self.normalizer.adapt(inputs)

  def call(self, inputs):
    # Stack the inputs
    inputs = self._stack(inputs)
    # Run them through all the layers.
    result = self.seq(inputs)

    return result

model = MyModel()

model.adapt(numeric_features_dict)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(numeric_features_dict, target_array, epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)

print(model.predict(dict(numeric_features.iloc[:3])))

inputs = {}
for name, column in numeric_features.items():
  inputs[name] = tf.keras.Input(
      shape=(1,), name=name, dtype=tf.float32)

print(inputs)

xs = [value for key, value in sorted(inputs.items())]

concat = tf.keras.layers.Concatenate(axis=1)
x = concat(xs)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.concatenate([value for key, value in sorted(numeric_features_dict.items())], axis=1))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True, show_layer_names=True)

model.fit(numeric_features_dict, target, epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)

binary_feature_names = ['sex', 'fbs', 'exang']

categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

inputs = {}
for name, column in df.items():
  if type(column[0]) == str:
    dtype = tf.string
  elif (name in categorical_feature_names or
        name in binary_feature_names):
    dtype = tf.int64
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

preprocessed = []

for name in binary_feature_names:
  inp = inputs[name]
  preprocessed.append(inp)

print(preprocessed)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.concatenate([value for key, value in sorted(numeric_features_dict.items())], axis=1))

numeric_inputs = []
for name in sorted(numeric_feature_names):
  numeric_inputs.append(inputs[name])

numeric_inputs = tf.keras.layers.Concatenate(axis=-1)(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

print(preprocessed)

vocab = ['a','b','c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup(['c','a','a','b','zzz']))

vocab = [1,4,7,99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

print(lookup([-1,4,1]))

for name in categorical_feature_names:
  vocab = sorted(set(df[name]))
  print(f'name: {name}')
  print(f'vocab: {vocab}\n')

  if type(vocab[0]) is str:
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
  else:
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

  x = inputs[name]
  x = lookup(x)
  preprocessed.append(x)

print(preprocessed)

preprocessed_result = tf.keras.layers.Concatenate(axis=1)(preprocessed)
print(preprocessed_result)

preprocessor = tf.keras.Model(inputs, preprocessed_result)

tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True,  show_layer_names=True)

print(preprocessor(dict(df.iloc[:1])))

body = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

print(inputs)

x = preprocessor(inputs)
print(x)

result = body(x)
print(result)

model = tf.keras.Model(inputs, result)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

x = {key: value.to_numpy() for key, value in dict(df).items()}
history = model.fit(x, target, epochs=5, batch_size=BATCH_SIZE)

ds = tf.data.Dataset.from_tensor_slices((
    dict(df),
    target
))

ds = ds.batch(BATCH_SIZE)

import pprint

for x, y in ds.take(1):
  pprint.pprint(x)
  print()
  print(y)

history = model.fit(ds, epochs=5)
