# netlify/functions/api.py

from flask import Flask, jsonify
from serverless_wsgi import handle  # Importante: el manejador de serverless-wsgi

# Crea la instancia de la aplicación Flask
app = Flask(__name__)

# Ejemplo de una ruta API
@app.route('/api/saludo', methods=['GET'])
def saludo():
    return jsonify({"mensaje": "¡Hola desde Flask en Netlify!"})

# Ejemplo de una ruta para la raíz
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Puedes servir una página principal o mostrar información
    return f"Ruta capturada: /{path}. La API está en /api/saludo"


# Esta es la función que Netlify ejecutará en cada petición
def handler(event, context):
    return handle(app, event, context)