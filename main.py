from load_data import get_data
from build_model import build_model
from train import train_model
from evaluate import evaluate_model
from predict import predict_species

def main():
    X_train, X_test, y_train, y_test, encoder = get_data()

    model = build_model()

    train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    predict_species(model, encoder, [5.1, 3.5, 1.4, 0.2])

if __name__ == "__main__":
    main()
