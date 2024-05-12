import tensorflow as tf
import os

class CNN():
    def __init__(self, train_dir, test_dir=None):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model = self.build_model(input_shape=(150, 150, 3))

    def preprocess(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=10,
            class_mode='categorical',
            color_mode='rgb'
        )
        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150, 150),
            batch_size=10,
            class_mode='categorical',
            color_mode='rgb'
        )
        return train_generator, validation_generator

    def build_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(5, activation='softmax'))
        return model

    def train(self, epochs):
        train_generator, validation_generator = self.preprocess()
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[CheckImprovement(), ApplyNeuroFuzzyLogic(self.train_dir,self.test_dir)]
        )
        return history

    def save_model(self, model_name):
        self.model.save(model_name)


class CheckImprovement(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CheckImprovement, self).__init__()
        self.prev_val_loss = float('inf')
        self.prev_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.prev_val_loss or logs['accuracy'] > self.prev_val_accuracy:
            self.prev_val_loss = logs['loss']
            self.prev_val_accuracy = logs['accuracy']
        else:
            self.model.stop_training = False


class ApplyNeuroFuzzyLogic(tf.keras.callbacks.Callback):
    def __init__(self, train_dir, test_dir):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.threshold = 0.5
        self.last_accuracy = None
        self.last_loss = None

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 1:
            acc_rate_of_change = (
                logs['accuracy'] - self.last_accuracy) / self.last_accuracy * 100
            loss_rate_of_change = (
                logs['loss'] - self.last_loss) / self.last_loss * 100

            predicted_probability = self.neuro_fuzzy_inference(
                acc_rate_of_change, loss_rate_of_change)

            if predicted_probability < self.threshold:
                self.model.stop_training = False

        self.last_accuracy = logs['accuracy']
        self.last_loss = logs['loss']

    def neuro_fuzzy_inference(self, acc_rate_of_change, loss_rate_of_change):
        acc_membership = self.membership_accuracy(acc_rate_of_change)
        loss_membership = self.membership_loss(loss_rate_of_change)

        rule1 = min(acc_membership['low'], loss_membership['low'])
        rule2 = min(acc_membership['medium'], loss_membership['medium'])
        rule3 = min(acc_membership['high'], loss_membership['high'])

        numerator = rule1 * 0.3 + rule2 * 0.6 + rule3 * 0.9
        denominator = rule1 + rule2 + rule3
        predicted_probability = numerator / denominator

        return predicted_probability

    def membership_accuracy(self, acc_rate_of_change):
        membership = {}
        if acc_rate_of_change < -50:
            membership['low'] = 1
            membership['medium'] = 0
            membership['high'] = 0
        elif acc_rate_of_change >= -50 and acc_rate_of_change <= 50:
            membership['low'] = 0
            membership['medium'] = 1
            membership['high'] = 0
        else:
            membership['low'] = 0
            membership['medium'] = 0
            membership['high'] = 1
        return membership

    def membership_loss(self, loss_rate_of_change):
        membership = {}
        if loss_rate_of_change < -50:
            membership['low'] = 1
            membership['medium'] = 0
            membership['high'] = 0
        elif loss_rate_of_change >= -50 and loss_rate_of_change <= 50:
            membership['low'] = 0
            membership['medium'] = 1
            membership['high'] = 0
        else:
            membership['low'] = 0
            membership['medium'] = 0
            membership['high'] = 1
        return membership

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
