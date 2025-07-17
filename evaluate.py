def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n📊 Dokładność na danych testowych: {accuracy:.2f}")
