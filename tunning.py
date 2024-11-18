import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar la clave API desde el archivo .env
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Ruta al archivo JSON
json_path = os.path.join("training_data", "sql_data.json")

# Cargar los datos de entrenamiento desde el archivo JSON
with open(json_path, "r") as f:
    training_data = json.load(f)

# Asegúrate de que `training_data` esté en el formato correcto: lista de diccionarios
# Ejemplo: [{"text_input": "1", "output": "2"}, ...]

# Configuración del modelo base y los parámetros de entrenamiento
base_model = "models/gemini-1.5-flash-001-tuning"
operation = genai.create_tuned_model(
    display_name="increment",
    source_model=base_model,
    epoch_count=20,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)

# Esperar hasta que el modelo esté listo
for status in operation.wait_bar():
    time.sleep(10)

# Obtener el resultado del entrenamiento
result = operation.result()
print(result)

# Crear el modelo ajustado y generar contenido
model = genai.GenerativeModel(model_name=result.name)
result = model.generate_content("III")
print(result.text)  # Ejemplo de salida
