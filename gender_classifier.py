from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Larger dataset
# [height, weight, shoe size]
X = [
    [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [170, 75, 42],
    [190, 90, 46], [175, 78, 42], [185, 85, 45], [169, 68, 41], [172, 74, 42],  # males
    [155, 50, 36], [160, 55, 37], [162, 53, 38], [158, 52, 36], [165, 60, 39],
    [170, 65, 40], [168, 59, 39], [159, 55, 37], [161, 58, 38], [166, 62, 39]   # females
]

Y = [
    'male','male','female','female','male',
    'male','male','male','male','male',
    'female','female','female','female','female',
    'female','female','female','female','female'
]

# Split into train/test (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Decision Tree": tree.DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear')
}

best_model = None
best_score = 0

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, Y_train)                # Train on training set
    predictions = model.predict(X_test)        # Predict on test set
    score = accuracy_score(Y_test, predictions) # Accuracy on test set

    print(f"{name} accuracy: {score:.2f}")

    if score > best_score:
        best_score = score
        best_model = name

print(f"\nBest model: {best_model} with accuracy {best_score:.2f}")
