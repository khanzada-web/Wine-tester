import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the dataset
wine_data = pd.read_csv("D:\\flask\\static\\winequality-red.csv")

# Preprocess the dataset
X = wine_data.drop('quality', axis=1)  # Features
y = wine_data['quality']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("D:\\flask\\module\\mymodel_linear_regression.pkl", "wb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/testing')
def testing():
    return render_template("testing.html")

@app.route('/visual')
def visual():
    return render_template("visual.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    input_features = [float(x) for x in request.form.values()]
    # Convert the features into a numpy array
    features = np.array(input_features).reshape(1, -1)
    # Make a prediction
    prediction = model.predict(features)[0]
    # Render the prediction on the template
    return render_template('testing.html', prediction_text="Predicted wine quality is {}".format(prediction))

def main():
    app.run()

if __name__ == "__main__":
    main()
