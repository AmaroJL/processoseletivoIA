import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    print("Carregando o dataset MNIST...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print("Construindo a arquitetura da CNN...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Iniciando o treinamento (3 épocas para execução rápida)...")
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

    print("\nAvaliando o modelo...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n✅ Acurácia final no conjunto de teste: {test_acc:.4f}")

    print("Salvando o modelo em formato .h5...")
    model.save('model.h5')
    print("Processo concluído com sucesso!")

if __name__ == '__main__':
    main()