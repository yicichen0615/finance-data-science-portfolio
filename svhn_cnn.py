import os
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

#set random seed so the result is more stable when I run the code repeatedly
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#set how many times the model goes through the training data.
EPOCHS = 20
#set how many images are used in one training step.
BATCH_SIZE = 128
#set learning rate
LEARNING_RATE = 0.001

#looking for data
def find_data_dir():
    possible_dirs = [
        os.path.join("datasets", "SVHN", "format2"),
        os.path.join("IndividualProject", "datasets", "SVHN", "format2"),
        os.path.join(".", "datasets", "SVHN", "format2"),
    ]
    #if both train and test files are found, return this folder.
    for folder in possible_dirs:
        train_path = os.path.join(folder, "train_32x32.mat")
        test_path = os.path.join(folder, "test_32x32.mat")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return folder
    #if data files cannot be found, stop the program
    raise FileNotFoundError(
        "Cannot find train_32x32.mat and test_32x32.mat. "
        "Please put this Python file next to the datasets folder."
    )


#clean SVHN data
def load_svhn_data(data_dir):
    train_file = os.path.join(data_dir, "train_32x32.mat")
    test_file = os.path.join(data_dir, "test_32x32.mat")

    train_data = sio.loadmat(train_file)
    test_data = sio.loadmat(test_file)
    #X is image data and y is label data.
    x_train = train_data["X"]
    y_train = train_data["y"]
    x_test = test_data["X"]
    y_test = test_data["y"]

    #original SVHN shape is (32, 32, 3, N). Keras needs (N, 32, 32, 3).
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    #change labels from shape (N, 1) to (N,).
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    #label 10 means digit 0.
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    #normalize pixel values from 0-255 to 0-1.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


#build cnn model
def build_model():
    #small L2 regularization
    weight_decay = 1e-4

    model = models.Sequential([
        #The input image size is 32 by 32 with 3 color channels.
        layers.Input(shape=(32, 32, 3)),
        #The first block learns simple image features like edges and colors.
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        #For second block, 
        #I use more filters here so the model can learn more digit patterns.
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),
        #Third block learns higher-level digit features.
        layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.40),
        #Flatten changes feature maps into one long vector.
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    #use sparse categorical crossentropy because labels are integer numbers
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_curves(history, output_dir):
    #saves accuracy and loss plots
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)
    #plot training accuracy and validation accuracy.
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training accuracy")
    plt.plot(epochs_range, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=200)
    plt.close()
    #Plot training loss and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Training loss")
    plt.plot(epochs_range, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()


def plot_sample_predictions(model, x_test, y_test, output_dir):
    #shows some test images and the model predictions
    #randomly choose 16 test images.
    index = np.random.choice(len(x_test), size=16, replace=False)
    images = x_test[index]
    true_labels = y_test[index]
    #predict labels for these images
    pred_labels = np.argmax(model.predict(images, verbose=0), axis=1)

    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title("T:" + str(true_labels[i]) + " P:" + str(pred_labels[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(model, x_test, y_test, output_dir):
    #makes and saves the confusion matrix
    #Predict the whole test set
    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    #Compare true labels and predicted labels
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    #Put numbers inside the matrix.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

#visualizes feature maps from some convolution layers
def plot_feature_maps(model, x_test, output_dir):
    # Use one image and show feature maps from three convolution layers
    one_image = x_test[0:1]
    #Find all Conv2D layers in the model.
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    #use three convolution layers for visualization
    selected_layers = [conv_layers[0], conv_layers[2], conv_layers[4]]

    for layer_number, layer in enumerate(selected_layers, start=1):
        #Build a small model to output the feature map of this layer.
        feature_model = models.Model(inputs=model.inputs, outputs=layer.output)
        feature_maps = feature_model.predict(one_image, verbose=0)
        #Only show 16 maps
        number_of_maps = min(16, feature_maps.shape[-1])
        plt.figure(figsize=(8, 8))
        for i in range(number_of_maps):
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap="gray")
            plt.axis("off")
        plt.suptitle("Feature maps from convolution layer " + str(layer_number))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_maps_layer_" + str(layer_number) + ".png"), dpi=200)
        plt.close()


def main():
    #Create output folder to save figures
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    #Find data and load it.
    data_dir = find_data_dir()
    print("Data folder:", data_dir)
    #Print shapes to check if the data is loaded correctly.
    x_train, y_train, x_test, y_test = load_svhn_data(data_dir)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    #build model
    model = build_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(
            #Reduce learning rate when validation loss stops improving.
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        EarlyStopping(
            #Stop training if validation accuracy does not improve.
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    #use 10% of the training data as validation data.
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    #test model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Final test loss:", test_loss)
    print("Final test accuracy:", test_accuracy)

    #Save figures for the report.
    plot_training_curves(history, output_dir)
    plot_sample_predictions(model, x_test, y_test, output_dir)
    plot_confusion_matrix(model, x_test, y_test, output_dir)

    print("All outputs are saved in the outputs folder.")


if __name__ == "__main__":
    main()
