# This research uses the following steps:
- 1. Collecting the historical time-series data of the monetary value of Aave, and the tweets with specific hashtags.
- 2. Data processing to organize the data into the format required by the model used in this study.
- 3. A combination of the llama2 model, LoRA method, and Alpaca-LoRA is used to analyze the price movements of the virtual currency.

Therefore, this study will first introduce the llama2 model, LoRA method, and Alpaca-LoRA, including the mathematical principles and advantages of using them. This study will also demonstrate the feasibility and innovativeness of this study by analyzing other literature and research in the field of virtual currency price analysis and prediction.

# llama2 model
The llama2 model is a model based on Google's transformer architecture, a self-attentive mechanism that is able to process input sequences while taking into account information from the global context, rather than relying solely on the local context. This property allows Llama2 to understand and generate complex linguistic structures and to perform better in natural language processing.
## Transformer architecture (Vaswani et al. 2017):
The architecture mainly contains the following features which ensure the advantages of the architecture when analyzing the global context.
1. Input Embeddings
- Position Encoding: Since the architecture does not have a loop structure to capture the positional information of the sequence, the architecture needs to add positional information to each input element through position encoding.
- Word Embeddings: maps each word in the input sequence to a high latitude vector space.
2. Multinomial Self-Attention Layer
- Self-attention mechanism: computes the context vector for each position of the input sequence, which in turn takes into account the information of the whole sequence. Attention weights are computed from three matrices: query, key and value, and these weights are weighted and averaged to generate a new representation.
- Multi-attention: In order to capture different contextual relationships, the self-attention mechanism is executed several times in parallel and the results are then stitched together.
3. Feed-Forward Neural Networks (FFNs):
- The output vector at each location passes through two linear layers with a nonlinear transformation between them using the ReLU activation function.
4. Residual Connections (RC):
- After each self-attention layer and feedforward neural network, a residual connection is added to help solve the problem of gradient vanishing in deep neural networks.
5. Layer Normalization:
- After the residual connections, a layer normalization operation is performed to stabilize the training process and accelerate convergence.
6. Encoder-Decoder structure:
- This architecture usually contains one or more encoder or decoder stacks. The encoder is responsible for extracting the semantic information of the input sequence and the decoder is used to generate the target sequence.
- All the layers of the encoder use the self-attention mechanism, while the first layer of the decoder uses the masked self-attention mechanism, which is designed to guarantee that the decoder cannot see future information, and the rest of the layers use the self-attention and the attention mechanism of the source sequence.
7. output Layer (Output Layer):
- The output of the decoder is passed through a linear layer and softmax function that generates a probability distribution of words for each position in the target sequence.
