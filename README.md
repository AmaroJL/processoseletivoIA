## 📝 Relatório do Candidato

👤 Identificação: **Amaro Júnior Silva Luna**


### 1️⃣ Resumo da Arquitetura do Modelo

Para resolver o problema de classificação dos dígitos manuscritos (dataset MNIST) respeitando as restrições de um ambiente de Edge AI, optei por uma arquitetura de Rede Neural Convolucional (CNN) enxuta e eficiente (Sequential). A estrutura foi desenhada da seguinte forma:

* **Camadas de Extração de Características:**
  * `Conv2D` (32 filtros, kernel 3x3, ativação ReLU) para detecção de bordas e padrões iniciais.
  * `MaxPooling2D` (pool size 2x2) para redução de dimensionalidade.
  * `Conv2D` (64 filtros, kernel 3x3, ativação ReLU) para abstração de padrões mais complexos.
  * `MaxPooling2D` (pool size 2x2) para nova compressão espacial.
* **Camadas de Classificação:**
  * `Flatten` para transformar a matriz bidimensional em um vetor unidimensional.
  * `Dense` oculta (64 neurônios, ativação ReLU).
  * `Dense` de saída (10 neurônios, ativação Softmax), representando a probabilidade de cada dígito (0 a 9).

Esta arquitetura possui um equilíbrio excelente entre baixo custo computacional e alta capacidade preditiva, ideal para dispositivos com recursos limitados.


### 2️⃣ Bibliotecas Utilizadas

* **TensorFlow / Keras (v2.x):** Utilizado como framework principal para importação do dataset, construção da arquitetura da rede, treinamento, avaliação e exportação (`.h5` e `.tflite`).
* **Python (Standard Library):** Para manipulação padrão de arquivos e execução dos scripts em ambiente de CI.


### 3️⃣ Técnica de Otimização do Modelo

A técnica aplicada no arquivo `optimize_model.py` foi a **Dynamic Range Quantization** (Quantização de Faixa Dinâmica). 

Através do `TFLiteConverter`, essa técnica otimiza o modelo convertendo estaticamente os pesos de ponto flutuante de 32 bits (float32) para números inteiros de 8 bits (int8) no momento da conversão, enquanto mantém as ativações em ponto flutuante. Isso reduz o tamanho do arquivo em quase 4 vezes, acelerando a inferência e reduzindo o consumo de memória, requisitos essenciais para deploy em sistemas embarcados e dispositivos Edge/IoT.


### 4️⃣ Resultados Obtidos

* **Treinamento e Acurácia:** O modelo foi treinado por 3 épocas (para garantir execução rápida na pipeline de CI) e atingiu uma acurácia no conjunto de testes de aproximadamente **99%**.
* **Deploy e Otimização:** O pipeline de salvamento funcionou corretamente. O modelo bruto foi salvo como `model.h5` e, em seguida, a quantização operou com sucesso, resultando no arquivo otimizado `model.tflite`, pronto para embarque.


### 5️⃣ Comentários Adicionais (Opcional)

* **Decisões Técnicas:** Limitar o número de épocas a 3 e utilizar no máximo duas camadas convolucionais foram decisões tomadas conscientemente para garantir que a GitHub Action do processo seletivo fosse executada rapidamente, sem estourar limites de timeout, respeitando as boas práticas de CI/CD.
* **Aprendizados:** O desafio ilustrou de forma clara o ciclo de vida real de um projeto de Machine Learning para a Indústria, onde buscar 100% de precisão muitas vezes é menos importante do que garantir que o modelo caiba e rode fluidamente no hardware final (Edge).