from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from itertools import combinations
import time

iris = datasets.load_iris()
X = iris.data
y = iris.target 

target_names = iris.target_names 

column_combinations = []
for r in range(1, len(X[0]) + 1):
   column_combinations.extend(combinations(range(len(X[0])), r))
   
for combo in column_combinations:
   X_subset = X[:, combo]
   X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
   clf = SVC(kernel='linear', C=1)
   start_time = time.time()
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   accuracy = (y_pred == y_test).mean()
   end_time = time.time()
   elapsed_time = end_time - start_time
   species_pred = [target_names[i] for i in y_pred]
   species_actual = [target_names[i] for i in y_test]
   
   print(f"Features: {combo}, Accuracy: {accuracy*100:.2f}")
   print(f"Time Taken: {elapsed_time:.4f} seconds\n")
   time.sleep(1.5)