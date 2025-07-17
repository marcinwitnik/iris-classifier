import numpy as np
import tensorflow as tf

def predict_species(model, encoder, input_data):
    input_array = np.array([input_data])  # musi być tablicą 2D
    prediction = model.predict(input_array)
    class_index = tf.argmax(prediction[0]).numpy()
    predicted_label = encoder.inverse_transform([class_index])
    print(f"🔍 Przewidywany gatunek: {predicted_label[0]}")
