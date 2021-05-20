from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle
from flask import jsonify

class Rna:

    def __init__(self):
        col_list = ['ENTIDAD_UM','SEXO','ENTIDAD_RES','NEUMONIA','EDAD','NACIONALIDAD','EMBARAZO','INDIGENA','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','CLASIFICACION_FINAL','TIPO_PACIENTE','FECHA_DEF','INTUBADO']

        df = pd.read_csv("30covid.csv", usecols=col_list)

        self.X=[]
        self.y=[]

        for i in range(len(df['FECHA_DEF'])):
            if(df['FECHA_DEF'][i]!="9999-99-99"):
                arrayX = []
                arrayX.append(df['SEXO'][i])
                arrayX.append(df['ENTIDAD_RES'][i])
                arrayX.append(df['NEUMONIA'][i])
                arrayX.append(df['EDAD'][i])
                arrayX.append(df['DIABETES'][i])
                arrayX.append(df['EPOC'][i])
                arrayX.append(df['ASMA'][i])
                arrayX.append(df['INMUSUPR'][i])
                arrayX.append(df['HIPERTENSION'][i])
                arrayX.append(df['OTRA_COM'][i])
                arrayX.append(df['CARDIOVASCULAR'][i])
                arrayX.append(df['OBESIDAD'][i])
                arrayX.append(df['RENAL_CRONICA'][i])
                arrayX.append(df['TABAQUISMO'][i])
                arrayX.append(df['OTRO_CASO'][i])
                arrayX.append(df['CLASIFICACION_FINAL'][i])
                self.X.append(arrayX)


                if(df['FECHA_DEF'][i]=="9999-99-99"):
                    self.y.append('1')
                else:
                    self.y.append('2')

        for i in range(len(self.y)):
            if(df['FECHA_DEF'][i]=="9999-99-99"):
                arrayX = []
                arrayX.append(df['SEXO'][i])
                arrayX.append(df['ENTIDAD_RES'][i])
                arrayX.append(df['NEUMONIA'][i])
                arrayX.append(df['EDAD'][i])
                arrayX.append(df['DIABETES'][i])
                arrayX.append(df['EPOC'][i])
                arrayX.append(df['ASMA'][i])
                arrayX.append(df['INMUSUPR'][i])
                arrayX.append(df['HIPERTENSION'][i])
                arrayX.append(df['OTRA_COM'][i])
                arrayX.append(df['CARDIOVASCULAR'][i])
                arrayX.append(df['OBESIDAD'][i])
                arrayX.append(df['RENAL_CRONICA'][i])
                arrayX.append(df['TABAQUISMO'][i])
                arrayX.append(df['OTRO_CASO'][i])
                arrayX.append(df['CLASIFICACION_FINAL'][i])
                self.X.append(arrayX)


                if(df['FECHA_DEF'][i]=="9999-99-99"):
                    self.y.append('1')
                else:
                    self.y.append('2')

    def fit(self):
        clf = MLPClassifier(solver='lbfgs',max_iter=10000 ,alpha=0.0001, hidden_layer_sizes=(16,8,4,2), random_state=0)
        clf.fit(self.X, self.y)
        self.pkl_filename = "pickle_model.pkl"
        with open(self.pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

    def predict(self, sexo, entidad_res, neumonia, edad, diabetes, epoc, asma, inmusupr, hipertension, otra_com, cardiovascular, obesidad, renal_cronica, tabaquismo, otro_caso, clasificacion_final):
        with open(self.pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        result = pickle_model.predict([[sexo, entidad_res, neumonia, edad, diabetes, epoc, asma, inmusupr, hipertension, otra_com, cardiovascular, obesidad, renal_cronica, tabaquismo, otro_caso, clasificacion_final]])
        return result[0]

