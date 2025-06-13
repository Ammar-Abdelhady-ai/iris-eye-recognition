import tensorflow as tf
from model_utils import save_model, save_encoder, update_user_dataframe
import joblib, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

cwd = os.getcwd()
LABEL_PATH = os.path.join(cwd, "labels.joblib")
ENCODER_PATH = os.path.join(cwd, "encoder.joblib")
COUNTER_PATH = os.path.join(cwd, "counter.joblib")
DF_PATH = os.path.join(cwd, "df_label.h5")




def fine_tune_model(model, input_image, label, name, address, birth_date, epochs=700):
    encoder = joblib.load(ENCODER_PATH)
    old_classes = list(encoder.classes_)
    if label in old_classes:
        print("ID already exists in encoder classes.")
        return "ID already exists in encoder classes."
    
    update_user_dataframe(name, address, label, birth_date)
    counter = joblib.load(COUNTER_PATH)
    old_classes[counter] = label

    encoder.fit(old_classes)
    encoded_label = encoder.transform([label])
    print(encoded_label)
    y_train = tf.keras.utils.to_categorical(encoded_label, 2000)

    joblib.dump(encoder, ENCODER_PATH)
    # Unfreeze last layers and train
    for layer in model.layers[:]:
        layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_image, y_train, epochs=epochs, verbose=1)

    save_model(model)
    save_encoder(encoder)
    counter += 1
    joblib.dump(counter, COUNTER_PATH)

    return "Model updated and retrained successfully."
