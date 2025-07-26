from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly
import json

app = Flask(__name__)

# Cargar modelo
model = joblib.load("model/model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    fig_json = None

        # Mostrar grafico
    df_demo = pd.DataFrame({
        "Variables": ["Age", "BMI", "blood_pressure", "glucose", "waist_size", "daily_steps", "stress_level", "cholesterol" ],
        "Valor Promedio": [42, 26.2, 128.98, 0.02, 0.32, 5, 4, 186.07]
    })
    fig = px.bar(df_demo, x="Variables", y="Valor Promedio", title="Variables promedio")
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    if request.method == "POST":
        try:
            features = [request.form[col] for col in ['age', 'bmi', 'blood_pressure', 'glucose','waist_size', 'daily_steps', 'stress_level', 'cholesterol' ]]
            df_pred = pd.DataFrame([features], columns=['age', 'bmi', 'blood_pressure', 'glucose','waist_size', 'daily_steps', 'stress_level', 'cholesterol' ])
            #features = [float(request.form[col]) for col in ['age', 'bmi', 'bp']]
            #df_pred = pd.DataFrame([features], columns=['age', 'bmi', 'bp'])
            pred = model.predict(df_pred)[0]
            if pred > 0:
                resultado = "Propenso a enfermedades"
            else:
                resultado = "Saludable"
        except Exception as e:
             resultado = f"Error en la predicción: {str(e)}"

        return render_template("index.html", resultado=resultado, fig=fig_json)

    return render_template("index.html", resultado=None, fig=None)

if __name__ == "__main__":
    #app.run(debug=True)
    import os
    port = int(os.environ.get("PORT", 5000))  # Render asigna un puerto autom·ticamente
    app.run(debug=False, host="0.0.0.0", port=port)