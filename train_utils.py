import tensorflow as tf
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model_utils import save_model

cwd = os.getcwd()

# Define file paths
LABEL_PATH = os.path.join(cwd, "labels.joblib")
ENCODER_PATH = os.path.join(cwd, "encoder.joblib")
COUNTER_PATH = os.path.join(cwd, "counter.joblib")
DF_PATH = os.path.join(cwd, "df_label.h5")

# Initialize files if they do not exist
if not os.path.exists(LABEL_PATH):
    joblib.dump([], LABEL_PATH)

if not os.path.exists(COUNTER_PATH):
    joblib.dump(0, COUNTER_PATH)

if os.path.exists(ENCODER_PATH):
    encoder = joblib.load(ENCODER_PATH)
else:
    encoder = LabelEncoder()

# Utility function to (re)fit the encoder and save it
def preprocess_labels(label_series):
    encoder = LabelEncoder()
    encoder.fit(label_series)
    joblib.dump(encoder, ENCODER_PATH)
    return encoder

def fine_tune_model(model, input_image, label, epochs=50):
    # Load or initialize label dataframe
    if os.path.exists(DF_PATH):
        df = pd.read_hdf(DF_PATH)
    else:
        df = pd.DataFrame(columns=['id', 'label'])

    # Check if label already exists
    if label in df['label'].values:
        return f"Label '{label}' already exists."

    # Load and update counter
    counter = joblib.load(COUNTER_PATH)
    new_row = pd.DataFrame({'id': [counter], 'label': [label]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_hdf(DF_PATH, key='df', mode='w')
    joblib.dump(counter + 1, COUNTER_PATH)

    # Encode labels
    encoder = preprocess_labels(df['label'])
    encoded_label = encoder.transform([label])[0]
    y_train = tf.keras.utils.to_categorical([encoded_label], num_classes=2000)

    # Fine-tune model
    for layer in model.layers[:-1]:
        layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_image, y_train, epochs=epochs, verbose=1)

    # Save the model and encoder
    save_model(model)
    joblib.dump(encoder, ENCODER_PATH)

    return "Model updated and retrained successfully."
