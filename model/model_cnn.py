import tensorflow as tf
from tensorflow.keras import layers,models


class ModelCNN:

    def __init__(self):
        pass

    def build_model(self):

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1,0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])


        model = tf.keras.Sequential([

            tf.keras.Input(shape=(110,110,1)),

            data_augmentation,

            layers.Conv2D(32,(3,3),activation='relu'),
            layers.MaxPooling2D(),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(),

            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(10, activation='softmax'),
        ])

        self.model = model


    def train_model(self, x_train, y_train, x_test, y_test):
        print(x_train.shape)
        print(x_train.dtype)
        print(x_train.min(), x_train.max())

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']
                           )


        history = self.model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=1000,
            batch_size=64
        )

        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Model Loss: {loss} Model Accuracy: {accuracy}")
