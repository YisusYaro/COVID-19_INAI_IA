from Rna import Rna
import flask
from flask import request, jsonify, render_template

rna = Rna()
rna.fit()

app = flask.Flask(__name__)
#app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return render_template('form.html')

@app.route('/predict', methods=['GET'])
def predict():
    return {'result' : str(rna.predict(request.args['sexo'], request.args['entidad_res'], request.args['neumonia'], request.args['edad'], request.args['diabetes'], request.args['epoc'], request.args['asma'], request.args['inmusupr'], request.args['hipertension'], request.args['otra_com'], request.args['cardiovascular'], request.args['obesidad'], request.args['renal_cronica'], request.args['tabaquismo'], request.args['otro_caso'], request.args['clasificacion_final']))}

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)
