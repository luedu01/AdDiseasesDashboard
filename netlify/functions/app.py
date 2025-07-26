from flask import Flask, render_template, request
from serverless_wsgi import handle 
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

    if request.method == "POST":
        try:
             features = [float(request.form[col]) for col in ['age', 'bmi', 'bp', 's6']]
             df_pred = pd.DataFrame([features], columns=['age', 'bmi', 'bp', 's6'])
             pred = model.predict(df_pred)[0]
             resultado = f"Riesgo estimado de diabetes: {pred:.4f}"
        except Exception as e:
             resultado = f"Error en la predicción: {str(e)}"

    # Mostrar grafico
    df_demo = pd.DataFrame({
        "Variable": ["Age", "BMI", "BP", "GLU"],
        "Valor Promedio": [0.03, 0.04, 0.05, 0.02]
    })
    fig = px.bar(df_demo, x="Variable", y="Valor Promedio", title="Variables promedio")
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("index.html", resultado=resultado, fig=fig_json)

# Esta es la función que Netlify ejecutará en cada petición
def handler(event, context):
    return handle(app, event, context)

#if __name__ == "__main__":
#    #app.run(debug=True)
#    import os
#    port = int(os.environ.get("PORT", 5000))  # Render asigna un puerto autom·ticamente
#    app.run(debug=False, host="0.0.0.0", port=port)
