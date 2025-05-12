from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from geopy.distance import distance
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Cargar el modelo multicategoría
model = load_model('recognizePlace_Categorical.h5')  # Tu modelo actualizado
IMG_SIZE = (224, 224)

# Coordenadas de referencia
REF_COORDS = {
    1: (17.020610, -96.721033),  # Lugar1
    2: (17.022546, -96.720905)   # Lugar2
}
RANGO_METROS = 10

# Información de las clases (nombre, descripción, video)
CLASES = {
    1: {
        "nombre": "Desconocido",
        "descripcion": "La imagen no corresponde a un lugar conocido.",
        "video": None
    },
    3: {
        "nombre": "Ocotlán de Morelos",
        "descripcion": "Un pueblo mágico en Oaxaca, conocido por su iglesia barroca y su mercado tradicional.",
        "video": "https://www.youtube.com/embed/v_Y0e69EeHQ?si=j4gj1lj8uQGqymm1"
    },
    2: {
        "nombre": "El Tule",
        "descripcion": "Hogar del famoso Árbol del Tule, un ahuehuete con el tronco más ancho del mundo.",
        "video": "https://www.youtube.com/embed/BSFHKaHJJRI?si=hTcbmHCadjyQY1qF"
    },
    0: {
        "nombre": "Cruz de Piedra",
        "descripcion": "Un lugar histórico en Oaxaca, conocido por su arquitectura colonial y su importancia cultural.",
        "video": "https://www.youtube.com/embed/1DJH6UqSlSU?si=gukOhRPxk-TrNGSO"
    }
}

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

    # Predicción (devuelve array de 3 probabilidades)
    predicciones = model.predict(img_array)[0]
    clase_predicha = np.argmax(predicciones)
    confianza = float(np.max(predicciones))
    nombre_clase = CLASES.get(clase_predicha, {"nombre": "Desconocido"})["nombre"]

    # Verificación de ubicación
    ubicacion_valida = False
    distancia_metros = None

    if clase_predicha in REF_COORDS:
        ref_coord = REF_COORDS[clase_predicha]
        distancia_metros = distance((lat, lon), ref_coord).meters
        ubicacion_valida = distancia_metros <= RANGO_METROS

    resultado_final = "Aceptado" if ubicacion_valida else "Rechazado"

    return jsonify({
        'clase_detectada': nombre_clase,
        'resultado': resultado_final,
        'confianza': round(confianza, 3),
        'distancia_metros': round(distancia_metros, 2) if distancia_metros is not None else None
    })

@app.route('/verificar-imagen', methods=['POST'])
def verificar_imagen():
    if 'imagen' not in request.files:
        return jsonify({'error': 'Falta el parámetro: imagen'}), 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo'}), 400

    try:
        imagen = load_img(BytesIO(archivo.read()), target_size=IMG_SIZE)
        img_array = img_to_array(imagen) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({'error': f'Error al procesar imagen: {e}'}), 400

    # Predicción (devuelve array de probabilidades)
    predicciones = model.predict(img_array)[0]
    clase_predicha = np.argmax(predicciones)
    confianza = float(np.max(predicciones))
    clase_info = CLASES.get(clase_predicha, {
        "nombre": "Desconocido",
        "descripcion": "No se pudo identificar la clase.",
        "video": None
    })

    return jsonify({
        'clase_detectada': clase_info["nombre"],
        'descripcion': clase_info["descripcion"],
        'video': clase_info["video"],
        'confianza': round(confianza, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)
