import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

res_size = 150
batch = 100

# Configure logical GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072) for _ in range(4)]
        )
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Available logical GPUs: {len(logical_gpus)}")

    except RuntimeError as e:
        print(f"Error in setting logical devices: {e}")

def train_flower_model(t_dir, v_dir, save_dir):
    strategy = tf.distribute.MirroredStrategy()
    
    num_replicas = strategy.num_replicas_in_sync
    print(f"Käytetään {num_replicas} kopiota")
    
    GBS = batch * num_replicas
    print(f"Global Batch size: {GBS}")

    t_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=t_dir,           # Corrected parameter here
        target_size=(res_size, res_size),
        batch_size=GBS,
        class_mode='sparse',
        shuffle=True
    )

    v_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=v_dir,           # Corrected parameter here
        target_size=(res_size, res_size),
        batch_size=GBS,
        class_mode='sparse',
        shuffle=False
    )

    print(f"Koulutusdatan määrä: {t_data_gen.samples}")

    # Define the model
    with strategy.scope():
        model = tf.keras.Sequential([
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(5, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    epochs = 10
    with open("train_log.out", "w") as log_file:
        log_file.write(f"Training started at {datetime.datetime.utcnow()}\n")
        
        history = model.fit(t_data_gen, validation_data=v_data_gen, epochs=epochs)

        for epoch in range(epochs):
            log_file.write(f"Epoch {epoch+1}: Loss: {history.history['loss'][epoch]}, "
                           f"Accuracy: {history.history['accuracy'][epoch]}, "
                           f"Validation Loss: {history.history['val_loss'][epoch]}, "
                           f"Validation Accuracy: {history.history['val_accuracy'][epoch]}\n")


    model_save_path = os.path.join(save_dir, "flower_model_TGu.keras")
    model.save(model_save_path)

    print("Training complete. Logs saved to train_log.out and model saved to ./models/flower_model_TGu.keras")

if __name__ == "__main__":
    train_dir = "/home/jovyan/shared/flower_photos/train"
    val_dir = "/home/jovyan/shared/flower_photos/val"
    save_dir = "/home/jovyan/ai-deep-learning-2/exercises/keras/"
    
    train_flower_model(train_dir, val_dir, save_dir)
