################################################################################
#                                  Import                                      #
################################################################################
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import os
from prm import (
    SYNTHMIX_DIR, MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MODEL_INPUT_SHAPE, MODEL_DENSE_UNITS, MODEL_NUM_CLASSES, BASE_DIR,
    SYNTHMIX_OUTPUT_SPEC, SYNTHMIX_OUTPUT_FRAC, MODEL_N_LAYERS, LEARNING_DECAY_RATE
)

################################################################################
#                                  User Settings                               #
################################################################################

# Working directory (use centralized param)
work_dir = BASE_DIR

# Input data (use names from prm)
folder_input = SYNTHMIX_DIR
file_name_x_in = SYNTHMIX_OUTPUT_SPEC
file_name_y_in = SYNTHMIX_OUTPUT_FRAC

# Output data
folder_output = MODEL_DIR
file_name_model = 'nn_model'



# Define model parameters outside the function
input_shape = MODEL_INPUT_SHAPE  # Number of bands
dense_units = MODEL_DENSE_UNITS  # Number of units/neurons/nodes in each dense layer
num_classes = MODEL_NUM_CLASSES  # Number of output classes
n_layers = MODEL_N_LAYERS
learning_decay_rate = LEARNING_DECAY_RATE

# Parameters for data and training
epochs = EPOCHS
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE

################################################################################
#                           Function Definitions                               #
################################################################################
def regression(input_shape, dense_units, num_classes, learning_rate, n_layers, learning_decay_rate, work_dir, folder_input, file_name_x_in, file_name_y_in, folder_output, file_name_model):
    # Normalizing the input to [0 - 1] by dividing by 10000.
    def norm(a):
        a_norm = a.astype(np.float32)
        a_norm = a_norm / 10000.
        return a_norm

    # ------------ Constructing Neural Network regression with Tensorflow
    def dense(a, num):
        layer = tf.keras.layers.Dense(num)(a)
        return layer

    # outdated model definition for reference 
    # def get_model(class_num):
    #     x_input = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    #     x = tf.nn.relu(dense(x_input, dense_units))
    #     x = tf.nn.relu(dense(x, dense_units))
    #     x = dense(x, class_num)
    #     model = tf.keras.Model(inputs=x_input, outputs=x)
    #     return model
    # 
    
    # Build model
    def get_model(input_shape, num_classes, n_layers, dense_units):
        x_in = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        x = tf.keras.layers.Flatten()(x_in)
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
        x_out = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs=x_in, outputs=x_out)
        return model

    #@tf.function
    #def get_loss(x, y, model, training):
    #    y_pred = model(x, training=training)
    #    loss = tf.keras.losses.MeanAbsoluteError()(y, y_pred)
    #    return loss
    
    @tf.function
    def get_loss(x, y, model, training):
        y_pred = model(x, training=training)
        mae = tf.keras.losses.MeanAbsoluteError()(y, y_pred)
        
        # Penalty when predicted fractions don't sum to 100
        sum_penalty = tf.reduce_mean(tf.abs(tf.reduce_sum(y_pred, axis=-1) - 1.0))
        
        return mae + 2 * sum_penalty  # tune the 0.1 weight

    @tf.function
    def train(x, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss(x, y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss


    # Training setup
    # Load data first to get iteration count for decay schedule
    x_in = norm(np.load(f'{folder_input}/{file_name_x_in}'))
    y_in = np.load(f'{folder_input}/{file_name_y_in}').astype(np.float32)
    iteration = int(y_in.shape[0] / batch_size)
    decay_steps = max(1, iteration)  # decay once per epoch
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=learning_decay_rate,
        staircase=True
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model =  get_model(input_shape, num_classes, n_layers, dense_units)


    # Shuffle training data
    train_list = list(range(y_in.shape[0]))
    random.shuffle(train_list)

    # Training
    for e in range(epochs):
        loss_train = 0
        for it in tqdm(range(iteration)):
            train_batch = train_list[it * batch_size: it * batch_size + batch_size]
            x_batch = x_in[train_batch, ...]
            y_batch = y_in[train_batch, ...]
            loss_train += train(x_batch, y_batch, model, opt)
        loss_train /= iteration
        print(f'Epoch {e}: MAE = {loss_train.numpy():.6f}')
        random.shuffle(train_list)
  

    # Save the trained model
    os.makedirs(os.path.join(folder_output), exist_ok=True)
    model.save(os.path.join(folder_output, file_name_model + '.keras'))

################################################################################
#                               Execution                                      #
################################################################################

def main():
    print('\n=== Script Execution Started ===')

    # Call the regression function with parameters
    regression(
        input_shape,
        dense_units,
        num_classes,
        learning_rate,
        n_layers,
        learning_decay_rate,
        work_dir,
        folder_input,
        file_name_x_in,
        file_name_y_in,
        folder_output,
        file_name_model
    )

    print('\n=== Script Execution Completed ===')


if __name__ == '__main__':
    main()
