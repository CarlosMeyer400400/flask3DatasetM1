from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta completa del archivo CSV
csv_path = 'periodontitis_dataset_70-30.csv'

# Intentar cargar datos desde el CSV
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')
except FileNotFoundError as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Verifica que los datos fueron cargados correctamente
if data is not None:
    # Convertir características categóricas en variables dummy
    data = pd.get_dummies(data, columns=['Sexo', 'Sangrado_Sondeo', 'Diabetes', 'Historial_Familiar', 'Consumo_Tabaco', 'Control_Placa'], drop_first=True)
    
    # Separar las características y la etiqueta
    X = data.drop('Tiene_Periodontitis', axis=1)
    y = data['Tiene_Periodontitis']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    app.logger.debug(f'Accuracy del modelo: {accuracy}')
    app.logger.debug(f'Informe de clasificación:\n{report}')
else:
    model = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model == None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    try:
        # Obtener los datos enviados en el request
        Edad = float(request.form['Edad'])
        Sexo = request.form['Sexo']
        Índice_Placa = float(request.form['Índice_Placa'])
        Profundidad_Bolsas = float(request.form['Profundidad_Bolsas'])
        Sangrado_Sondeo = request.form['Sangrado_Sondeo']
        Pérdida_Inserción = float(request.form['Pérdida_Inserción'])
        Diabetes = request.form['Diabetes']
        Historial_Familiar = request.form['Historial_Familiar']
        Higiene_Bucal = int(request.form['Higiene_Bucal'])
        Consumo_Tabaco = request.form['Consumo_Tabaco']
        Control_Placa = request.form['Control_Placa']
        
        # Convertir a DataFrame y crear dummies
        data_df = pd.DataFrame([[Edad, Sexo, Índice_Placa, Profundidad_Bolsas, Sangrado_Sondeo, Pérdida_Inserción, Diabetes, Historial_Familiar, Higiene_Bucal, Consumo_Tabaco, Control_Placa]],
                               columns=['Edad', 'Sexo', 'Índice_Placa', 'Profundidad_Bolsas', 'Sangrado_Sondeo', 'Pérdida_Inserción', 'Diabetes', 'Historial_Familiar', 'Higiene_Bucal', 'Consumo_Tabaco', 'Control_Placa'])
        data_df = pd.get_dummies(data_df, columns=['Sexo', 'Sangrado_Sondeo', 'Diabetes', 'Historial_Familiar', 'Consumo_Tabaco', 'Control_Placa'], drop_first=True)
        
        # Asegurarse de que las columnas coincidan con las del modelo
        data_df = data_df.reindex(columns=X_train.columns, fill_value=0)
        
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'Tiene_Periodontitis': "Sí" if prediction[0] == 1 else "No"})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
