import tensorflow as tf
import os 

class CNN():
    def __init__(self, train_dir,test_dir):
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
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary',
            color_mode='rgb'
        )
        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary',
            color_mode='rgb'
        )
        return train_generator, validation_generator
    
    def build_model(self,input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    
    def train(self,epochs):
        train_generator, validation_generator = self.preprocess()
        self.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        history = self.model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=50
        )
        return history
    
    def save_model(self,model_name):
        self.model.save(model_name)