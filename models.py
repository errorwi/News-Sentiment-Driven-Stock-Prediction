from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_models(X_train, y_train, X_test, y_test):
    models = {
        "LogReg": LogisticRegression(),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc)

    return results
