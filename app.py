from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from geopy.geocoders import Nominatim
from geopy.distance import distance
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Coordenadas de referencia
REF_COORD = (17.021448, -96.721127)
RANGO_METROS = 10  # Rango permitido en metros
            
# Cargar el modelo entrenado
model = load_model('modelo_lugar.h5')
IMG_SIZE = (224, 224)

@app.route('/verificar-lugar', methods=['POST'])
def verificar_lugar():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se proporcionó imagen'}), 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo'}), 400

    try:
        # Convertir el archivo en stream y cargar imagen
        imagen = load_img(BytesIO(archivo.read()), target_size=IMG_SIZE)
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {e}'}), 400

    # Preprocesar imagen
    img_array = img_to_array(imagen) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir
    pred = model.predict(img_array)[0][0]
    resultado = "Es el lugar correcto" if pred > 0.5 else "No es el lugar"

    return jsonify({'resultado': resultado, 'score': float(pred)})

@app.route('/verificar-ubicacion', methods=['GET'])
def verificar_ubicacion():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Debe proporcionar los parámetros lat y lon'}), 400
    
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({'error': 'Los parámetros lat y lon deben ser numéricos'}), 400

    # Coordenadas del usuario
    user_coord = (lat, lon)
    
    # Calcular la distancia en metros
    dist = distance(REF_COORD, user_coord).meters

    if dist <= RANGO_METROS:
        return jsonify({
            'mensaje': 'La ubicación está dentro del rango permitido',
            'distancia': dist
        })
    else:
        return jsonify({
            'mensaje': 'La ubicación está fuera del rango permitido',
            'distancia': dist
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
