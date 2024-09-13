import os
import tensorflow as tf
import time
import datetime
from tensorflow.keras.models import load_model

def test_flower_model(test_dir):
    model = load_model("/home/jovyan/ai-deep-learning-2/exercises/keras/flower_model_NN.keras")

    test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=test_dir,
        target_size=(150, 150),
        batch_size=32,
        shuffle=False,
        class_mode = 'sparse'
    )

    start_time = time.time()
    
    loss, accuracy = model.evaluate(test_data_gen)
    
    tested_images = test_data_gen.samples
    correct_percentage = accuracy * 100
    test_loss_per_image = loss / tested_images

    end_time = time.time()
    run_time = end_time - start_time

    log_file = open("test_log.out", "w")
    log_file.write(f"Testi alkoi {datetime.datetime.utcnow()}\n")
    log_file.write(f"Aika: {run_time:.2f} sekuntia\n")
    log_file.write(f"Testattiin kuvia: {tested_images}\n")
    log_file.write(f"Oikein: {correct_percentage:.2f}%\n")
    log_file.write(f"Test loss kuvaa kohden: {test_loss_per_image:.5f}\n")
    log_file.close()

    print(f"Testi valmis. Logit tallennettu test_log.out")
    
    return tested_images, correct_percentage, test_loss_per_image

if __name__ == "__main__":
    test_dir = "/home/jovyan/shared/flower_photos/test"
    
    
    test_flower_model(test_dir)
