import tensorflow as tf

def main():
    print("Carregando o modelo original (model.h5)...")
    model = tf.keras.models.load_model('model.h5')

    print("Iniciando a conversão para TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    print("Aplicando Dynamic Range Quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter.convert()

    print("Salvando o modelo otimizado (model.tflite)...")
    with open('model.tflite', 'wb') as f:
        f.write(tflite_quant_model)
        
    print("✅ Modelo convertido e otimizado com sucesso para Edge AI!")

if __name__ == '__main__':
    main()