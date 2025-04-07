import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from geopy.distance import distance
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/verificar-lugar": {"methods": ["POST"]}})

# Cargar el modelo multicategoría
MODEL_PATH = "recognizePlace.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El modelo {MODEL_PATH} no se encuentra en el servidor.")

model = load_model(MODEL_PATH)  # Cargar el modelo entrenado
IMG_SIZE = (224, 224)

# Coordenadas de referencia
REF_COORDS = {
    1: (17.020610, -96.721033),  # Lugar1
    2: (17.022546, -96.720905)   # Lugar2
}
RANGO_METROS = 25

# Nombres de las clases (asegúrate de que el orden corresponda con el entrenamiento)
CLASES = {
    0: "Inválido",
    1: "San Juan Bautista de La Salle",
    2: "Puerta de Ingenierías"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"mensaje": "API funcionando correctamente"}), 200

@app.route('/verificar-lugar', methods=['POST'])
def verificar_lugar():
    if 'imagen' not in request.files or 'lat' not in request.form or 'lon' not in request.form:
        return jsonify({'error': 'Faltan parámetros: imagen, lat o lon'}), 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo'}), 400

    try:
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
    except ValueError:
        return jsonify({'error': 'Lat y Lon deben ser numéricos'}), 400

    try:
        imagen = load_img(BytesIO(archivo.read()), target_size=IMG_SIZE)
        img_array = img_to_array(imagen) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({'error': f'Error al procesar imagen: {e}'}), 400

    # Predicción
    predicciones = model.predict(img_array)[0]
    clase_predicha = np.argmax(predicciones)
    confianza = float(np.max(predicciones))
    nombre_clase = CLASES.get(clase_predicha, "Desconocido")

    # Verificación de ubicación
    ubicacion_valida = False
    distancia_metros = None

    if clase_predicha in REF_COORDS:
        ref_coord = REF_COORDS[clase_predicha]
        distancia_metros = distance((lat, lon), ref_coord).meters
        ubicacion_valida = distancia_metros <= RANGO_METROS

    # Evaluación de estado
    imagen_valida = clase_predicha in REF_COORDS

    if imagen_valida and ubicacion_valida:
        estado = 2
        mensaje = "Foto y coordenadas correctas"
        resultado_final = "Aceptado"
    elif not imagen_valida and ubicacion_valida:
        estado = 1
        mensaje = "Coordenadas correctas, pero la foto no coincide"
        resultado_final = "Rechazado"
    elif imagen_valida and not ubicacion_valida:
        estado = 0
        mensaje = "Foto correcta, pero las coordenadas no coinciden"
        resultado_final = "Rechazado"
    else:
        estado = -1
        mensaje = "Foto y coordenadas incorrectas"
        resultado_final = "Rechazado"

    return jsonify({
        'clase_detectada': nombre_clase,
        'resultado': resultado_final,
        'confianza': round(confianza, 3),
        'distancia_metros': round(distancia_metros, 2) if distancia_metros is not None else None,
        'estado': estado,
        'mensaje_detallado': mensaje
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Obtiene el puerto de Railway
    app.run(host='0.0.0.0', port=port, debug=True)
