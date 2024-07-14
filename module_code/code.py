import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


data_set = pd.read_csv('D:\\flask\\static\\winequality-red.csv')
print(data_set.info())


data_set = data_set.dropna()  


features = data_set.drop('quality', axis=1)
target = data_set['quality']


data_set.hist(bins=30, figsize=(15, 12), layout=(4, 3))
plt.suptitle('Histograms of All Features', fontsize=20)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.8)
plt.show()


plt.figure(figsize=(15, 10))
sns.boxplot(data=data_set)
plt.title('Box Plots of All Features', fontsize=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = data_set.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix', fontsize=20)
plt.tight_layout()
plt.show()

sns.pairplot(data_set)
plt.suptitle('Scatter Plot Matrix', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


features['sulphates_acidity_interaction'] = features['sulphates'] * features['fixed acidity']


X_Train, X_Test, Y_Train, Y_Test = train_test_split(features, target, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_Train, Y_Train)


Y_Pred = model.predict(X_Test)


plt.figure(figsize=(10, 6))
plt.scatter(X_Test['pH'], Y_Test, label='True Values', alpha=0.7)
plt.scatter(X_Test['pH'], Y_Pred, color='red', label='Predicted Values', alpha=0.5)
plt.xlabel('pH', fontsize=20)
plt.ylabel('Quality', fontsize=20)
plt.title('True vs Predicted Quality (pH Feature)', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()


mse = mean_squared_error(Y_Test, Y_Pred)
print(f'Mean Squared Error: {mse:.2f}')


train_sizes, train_scores, test_scores = learning_curve(
    model, X_Train, Y_Train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Error')
plt.plot(train_sizes, test_scores_mean, label='Validation Error')
plt.xlabel('Training Size', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.legend()
plt.title('Learning Curves', fontsize=20)
plt.tight_layout()
plt.show()

median_quality = target.median()
Y_Test_bin = (Y_Test > median_quality).astype(int)
Y_Pred_bin = (Y_Pred > median_quality).astype(int)


cm = confusion_matrix(Y_Test_bin, Y_Pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix', fontsize=20)
plt.tight_layout()
plt.show()


# pickle.dump(mymodel, open("D:\\flask\\module\\mymodel_knn.pkl","wb"))
# print("Sucess")
