from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle
from random import *
import flask

col_list = ['ENTIDAD_UM','SEXO','ENTIDAD_RES','NEUMONIA','EDAD','NACIONALIDAD','EMBARAZO','INDIGENA','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','TABAQUISMO','OTRO_CASO','CLASIFICACION_FINAL','TIPO_PACIENTE','FECHA_DEF','INTUBADO']

df = pd.read_csv("30covid.csv", usecols=col_list)



X=[]
y=[]




for i in range(len(df['FECHA_DEF'])):
    if(df['FECHA_DEF'][i]!="9999-99-99"):
        arrayX = []
        
        #arrayX.append(df['ENTIDAD_UM'][i])
        arrayX.append(df['SEXO'][i])
        arrayX.append(df['ENTIDAD_RES'][i])
        arrayX.append(df['NEUMONIA'][i])
        arrayX.append(df['EDAD'][i])
        #arrayX.append(df['NACIONALIDAD'][i])
        #arrayX.append(df['EMBARAZO'][i])
        #arrayX.append(df['INDIGENA'][i])
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
        X.append(arrayX)


        if(df['FECHA_DEF'][i]=="9999-99-99"):
            y.append('1')
        else:
            y.append('2')
            print(i)
    

print(len(X))
print(len(y))

for i in range(len(y)):
    if(df['FECHA_DEF'][i]=="9999-99-99"):
        arrayX = []
        
        #arrayX.append(df['ENTIDAD_UM'][i])
        arrayX.append(df['SEXO'][i])
        arrayX.append(df['ENTIDAD_RES'][i])
        arrayX.append(df['NEUMONIA'][i])
        arrayX.append(df['EDAD'][i])
        #arrayX.append(df['NACIONALIDAD'][i])
        #arrayX.append(df['EMBARAZO'][i])
        #arrayX.append(df['INDIGENA'][i])
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
        X.append(arrayX)


        if(df['FECHA_DEF'][i]=="9999-99-99"):
            y.append('1')
        else:
            y.append('2')
            print(i)

print(X)
print(y)

clf = MLPClassifier(solver='lbfgs',max_iter=10000 ,alpha=0.0001, hidden_layer_sizes=(16,8,4,2), random_state=0)

print(clf.fit(X, y))

# Save to file in the current working directory
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)




print("prediccion")

result = pickle_model.predict([[2, 8, 2, 53, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7]])

print(result)