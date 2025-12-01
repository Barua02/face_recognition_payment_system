"""
train_evaluate.py
Train, evaluate, and save the CNN model for face recognition.
"""
from data_preprocessing import load_and_preprocess
from model_building import CNN_classification_model
from tensorflow.keras.models import save_model

def train_and_evaluate():
    x_train, x_test, y_train, y_test, _ = load_and_preprocess()
    input_shape = x_train.shape[1:]
    output_size = y_train.shape[1]
    model = CNN_classification_model(input_shape, output_size)
    model.summary()
    history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    model.save('./final_faceReco.h5')
    print("Saved complete model to './final_faceReco.h5'")
    model.save_weights('./faceReco.weights.h5')
    print("Saved model weights to './faceReco.weights.h5'")

if __name__ == "__main__":
    train_and_evaluate()
