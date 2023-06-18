# importing required libraies
import pandas as pd
import numpy as np


## Data Ingestions step
data=pd.read_csv('notebooks/data/aad.csv')
data.head()


## making a copy of original dataset so that it do not harm our original dataset.
df = data.copy()
df.head()



## Lets drop the  unnamed column
df=df.drop(labels=['Unnamed: 0'],axis=1)
df.head()


#dropping the duplicated values
df = df.drop_duplicates()



## segregate numerical and categorical columns
numerical_columns=df.columns[df.dtypes!='object']
categorical_columns=df.columns[df.dtypes=='object']



## correlation
import seaborn as sns
sns.heatmap(df[numerical_columns].corr(),annot=True)


## Independent and dependent features
X = df.drop(labels=['Sales'],axis=1)
Y = df[['Sales']]



# Define which columns should be encode and which should be scaled
numerical_cols = X.select_dtypes(exclude='object').columns


from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


## Numerical Pipeline
num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
    ])
preprocessor=ColumnTransformer([
('num_pipeline',num_pipeline,numerical_cols)
])


## Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=30)



X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())



## Model Training

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


model = LinearRegression()
model.fit(X_train,y_train)


## Making Predictions
prediction = model.predict(X_test)


## Calculating the accuracy
r2_square = r2_score(y_test , prediction)
print("R2 score:", r2_square)



X_new = np.array([[  5.3, 2.5, 1.2 ],[3, 2, 1], [  4.9, 2.2, 3.8], [  5.3, 2.5, 4.6 ]])
#Prediction of the species from the input vector
new_prediction = model.predict(X_new)
print("Prediction of Sales: {}".format(new_prediction))



# Save the model
import pickle
with open('regression.pickle', 'wb') as f:
    pickle.dump(model, f)



# Load the model
with open('regression.pickle', 'rb') as f:
    model = pickle.load(f)


model.predict(X_new)