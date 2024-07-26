from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos
ordinal_encoder = joblib.load('modelo_ordinalEncoder_Den.pkl')
scaler = joblib.load('modelo_Scaler_Den.pkl')
modelRF = joblib.load('modelo_RandomForest_Den.pkl')

app.logger.debug('4 modelos cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
       # Obtener los datos del formulario
        Indice_Placa_Dental = request.form['Indice_Placa_Dental']
        Sangrado_Sondeo = request.form['Sangrado_Sondeo']
        Perdida_Insercion_Clinica = request.form['Perdida_Insercion_Clinica']
        Control_Placa = request.form['Control_Placa']
        Diabetes = request.form['Diabetes']
        Historial_Familiar = request.form['Historial_Familiar']
        Higiene_Bucal = request.form['Higiene_Bucal']


        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[Indice_Placa_Dental, Sangrado_Sondeo, Perdida_Insercion_Clinica, Control_Placa, Diabetes, Historial_Familiar, Higiene_Bucal]], 
                                columns=['Indice_Placa_Dental', 'Sangrado_Sondeo', 'Perdida_Insercion_Clinica', 'Control_Placa', 'Diabetes', 'Historial_Familiar', 'Higiene_Bucal'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Seleccionar las columnas categóricas a convertir a numérico
        categorical_columns = ['Sangrado_Sondeo', 'Diabetes', 'Historial_Familiar']
# Transformar las columnas categóricas a numérico
        data_df[categorical_columns] = ordinal_encoder.transform(data_df[categorical_columns])
        app.logger.debug(f'Datos transformados a numérico: {data_df}')

        # Escalar los datos
        scaler_df = scaler.transform(data_df)
        scaler_df = pd.DataFrame(scaler_df, columns=data_df.columns)
        app.logger.debug(f'DataFrame escalado: {scaler_df}')

        
        # Realizar la predicción
        prediction = modelRF.predict(scaler_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Convertir la predicción a un tipo de datos serializable (int)
        prediction_serializable = int(prediction[0])

        # Mapear la predicción a una categoría, ajusta según las categorías de tu modelo
        if prediction_serializable == 0:
            category = "No tiene Periodontitis"
        elif prediction_serializable == 1:
            category = "Sí tiene Periodontitis"
       
        else:
            category = "Unknown"

        # Devolver la predicción como respuesta JSON
        return jsonify({'categoria': category})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
