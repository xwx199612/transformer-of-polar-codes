import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split

import transformer as trans

class LossMetricsLogger(Callback):
    def __init__(self, log_batch, log_epoch):
        super(LossMetricsLogger, self).__init__()
        self.log_batch = log_batch
        self.log_epoch = log_epoch
        self.loss_history = []
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f'Epoch {epoch+1}:\n')
        
    def on_epoch_end(self, epoch, logs=None):        
        epoch_loss = logs['loss']
        epoch_metric = logs['accuracy']  # 假设您想要保存准确率
        self.loss_history.append({'epoch': epoch, 'loss': epoch_loss, 'metric': epoch_metric})
        df = pd.DataFrame(self.loss_history)
        df.to_csv(self.log_file, index=False)
        
    ##def on_batch_begin(self, batch, logs=None):
    
    def on_batch_end(self, batch, logs=None):
        print(f'batch {batch+1} - loss: {logs["loss"]:.4f}\n')
        batch_loss = logs['loss']
        batch_metric = logs['accuracy']  # 假设您想要保存准确率
        self.loss_history.append({'batch': batch, 'loss': batch_loss, 'metric': batch_metric})
        df = pd.DataFrame(self.loss_history)
        df.to_csv(self.log_file, index=False, mode='a', header=False)


def Training_Parameter_Choosing(learning_rate, epochs, batch_size, train_data, val_data):

    model = trans.Transformer(
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 定義損失函數和優化器
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.CategoricalCrossentropy(targets, predictions)

    grad = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(grad, transformer.trainable_variables))


    model.compile(optimizer=optimizer, loss=loss_fn)

    model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data)

    return model.evaluate(val_data)

def Model_Parameter_Choosing(learning_rate, num_layers, d_model, num_heads, d_ff, train_data, val_data):

    model = trans.Transformer(
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 定義損失函數和優化器
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.CategoricalCrossentropy(targets, predictions)

    grad = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(grad, transformer.trainable_variables))


    model.compile(optimizer=optimizer, loss=loss_fn)

    model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data)

    return model.evaluate(val_data)
    



##num_layers = 4
##d_model = 128
##num_heads = 8
##dff = 512
##input_vocab_size = 10000
##target_vocab_size = 8000
##pe_input = 1000
##pe_target = 1200


# 定義超參數空間
training_parameter_space = {
    'learning_rate' : [0.001, 0.01, 0.1],
    'epochs' : [10, 20, 30],
    'batch_size' : [16, 32, 64],

}
# 定義超參數空間
model_parameter_space = {
    'num_layers' : [2, 4, 6, 8],
    'd_model' : [32, 64, 128],
    'num_heads' : [4,8],
    'dff' : [128, 256, 512],
}

# 進行隨機搜索
num_samples = 10
param_sampler = ParameterSampler(param_space, n_iter=num_samples, random_state=0)

best_model = None
best_loss = float('inf')

# data preprocessing
X = np.loadtxt('X.txt', dtype=float)        ## input
Y = np.loadtxt('Y.txt', dtype=int)          ## input
Ysoft = np.loadtxt('Ysoft.txt', dtype=float)## input
FLIP = np.loadtxt('FLIP.txt', dtype=int)    ## target
    
X_train, X_test, Y_train, Y_test, Ysoft_train, Ysoft_test, FLIP_train, FLIP_test = train_test_split(X, Y, Ysoft, FLIP, test_size=0.2, random_state=42)


# 訓練迴圈
epochs = 10
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (inputs, targets) in enumerate(train_dataset):
        train_step(inputs, targets)

    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')
    
for params in param_sampler:
    ##learning_rate = params['learning_rate']
    ##hidden_units = params['hidden_units']
    ##epochs = params['epochs']
    ##batch_size = params['batch_size']
    num_layers = params['num_layers']
    d_model = params['d_model']
    num_heads = params['num_heads']
    dff = params['dff']

    model = Training_Parameter_Choosing(learning_rate, hidden_units, epochs, batch_size, (X_train, Y_train, Ysoft_train, FLIP_train), (X_test, Y_test, Ysoft_test, FLIP_test))
    val_loss = model[0]

    if val_loss < best_loss:
        best_model = model
        best_loss = val_loss

print("Best Validation Loss:", best_loss)
print("Best Hyperparameters:", best_model.get_config())