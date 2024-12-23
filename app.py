from flask import jsonify, Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

# Inicializar la app Flask y habilitar CORS
app = Flask(__name__)
CORS(app)  

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="modelo_diatraea_vgg16_augmented.tflite")
interpreter.allocate_tensors()

# Obtener las entradas y salidas del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir las clases según el modelo
CLASS_NAMES = ["Adulta", "Con daño", "Huevos", "Larvas", "Sin daño", "Cogollero"]

# Función para procesar la imagen
def preprocess_image(image):
    image = image.resize((150, 150))  # Ajustar al tamaño de entrada del modelo
    image = np.array(image)  # Convertir a array de numpy
    image = np.expand_dims(image, axis=0)  # Añadir dimensión para el lote
    image = image / 255.0  # Normalizar valores entre 0 y 1
    return image

# Ruta para hacer predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No se encontró un archivo."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No se seleccionó un archivo."}), 400

    # Abrir y preprocesar la imagen
    image = Image.open(file.stream)  # Usar file.stream para evitar guardar la imagen en uploads
    processed_image = preprocess_image(image)

    # Realizar la predicción usando TensorFlow Lite
    input_data = np.array(processed_image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtener la predicción
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    predicted_probability = output_data[0][predicted_class_index]

    # Crear un directorio para la clase si no existe
    class_dir = os.path.join("predictions", predicted_class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Guardar la imagen en el directorio correspondiente
    final_filename = os.path.join(class_dir, f"{secure_filename(file.filename)}")
    image.save(final_filename)

    # Responder con el resultado
    return jsonify({
        "success": True,
        "class": predicted_class_name,
        "probability": float(predicted_probability),
        "saved_image": final_filename  # Incluir la ruta de la imagen guardada
    }), 200


# Ruta para verificar el estado de la API
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "success": True,
        "message": "La API está activa y funcional."
    }), 200

# Ruta para obtener métricas del modelo
@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_info = {
        "success": True,
        "model_name": "modelo_diatraea_vgg16_augmented",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "description": "Modelo para clasificación de plagas en 6 categorías."
    }
    return jsonify(metrics_info), 200

# Iniciar la app Flask
if __name__ == '__main__':
    app.run(debug=True)
