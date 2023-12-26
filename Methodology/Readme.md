# This research uses the following steps:
- 1. Collecting the historical time-series data of the monetary value of Aave, and the tweets with specific hashtags.
- 2. Data processing to organize the data into the format required by the model used in this study.
- 3. A combination of the llama2 model, LoRA method, and Alpaca-LoRA is used to analyze the price movements of the virtual currency.

Therefore, this study will first introduce the llama2 model, LoRA method, and Alpaca-LoRA, including the mathematical principles and advantages of using them. This study will also demonstrate the feasibility and innovativeness of this study by analyzing other literature and research in the field of virtual currency price analysis and prediction.

Then we will introfuce the data process step, which shows how data gathered from online resources can be processed to fit the requirements of the model.

# Llama2 model
The llama2 model is a model based on Google's transformer architecture, a self-attentive mechanism that is able to process input sequences while taking into account information from the global context, rather than relying solely on the local context. This property allows Llama2 to understand and generate complex linguistic structures and to perform better in natural language processing.
## Transformer architecture (Vaswani et al. 2017):
The architecture mainly contains the following features which ensure the advantages of the architecture when analyzing the global context.
#### 1. Input Embeddings
- Position Encoding: Since the architecture does not have a loop structure to capture the positional information of the sequence, the architecture needs to add positional information to each input element through position encoding.
- Word Embeddings: maps each word in the input sequence to a high latitude vector space.
#### 2. Multinomial Self-Attention Layer
- Self-attention mechanism: computes the context vector for each position of the input sequence, which in turn takes into account the information of the whole sequence. Attention weights are computed from three matrices: query, key and value, and these weights are weighted and averaged to generate a new representation.
- Multi-attention: In order to capture different contextual relationships, the self-attention mechanism is executed several times in parallel and the results are then stitched together.
#### 3. Feed-Forward Neural Networks (FFNs):
- The output vector at each location passes through two linear layers with a nonlinear transformation between them using the ReLU activation function.
#### 4. Residual Connections (RC):
- After each self-attention layer and feedforward neural network, a residual connection is added to help solve the problem of gradient vanishing in deep neural networks.
#### 5. Layer Normalization:
- After the residual connections, a layer normalization operation is performed to stabilize the training process and accelerate convergence.
#### 6. Encoder-Decoder structure:
- This architecture usually contains one or more encoder or decoder stacks. The encoder is responsible for extracting the semantic information of the input sequence and the decoder is used to generate the target sequence.
- All the layers of the encoder use the self-attention mechanism, while the first layer of the decoder uses the masked self-attention mechanism, which is designed to guarantee that the decoder cannot see future information, and the rest of the layers use the self-attention and the attention mechanism of the source sequence.
#### 7. output Layer (Output Layer):
- The output of the decoder is passed through a linear layer and softmax function that generates a probability distribution of words for each position in the target sequence.

The advantage of the Transformer architecture is mainly the ability to ensure parallelization, that is, the ability to process entire sequences at the same time. This is reflected in the ability to recalculate the weights and change the overall result for each piece of information fed into the model. These features make the architecture excellent for natural language processing such as text generation and question and answer systems.

### Advantages of the Llama2 model:
- Open source: as an open source model, Llama 2 allows researchers and developers to freely use, modify, and optimize the model, which is very beneficial for driving innovation and application development.
- Large-scale training data: Large models are usually trained based on a large amount of training data, which helps to improve the model's generalization ability and recognition of complex patterns.
- Multi-tasking: Large models such as Llama 2 usually have strong transfer learning capabilities and are able to share knowledge and patterns across multiple tasks, thus achieving good performance on new tasks faster.

### Existing studies
Currently we are not able to find papers related to financial analysis using the Llama2 model in online repositories such as Google scholar, and there are cases of using other models in areas related to financial analysis using large language models, which will be covered in the literature.

<img src="Llama2 model.png" alt="Llama2 model">

# LoRA method
The LoRA approach is based on efficient and precise control of parameter updates for large pre-trained models. The core idea of this method is to represent the changes of model parameters through a low-rank matrix instead of directly updating all parameters in the model. In the fine-tuning process, LoRA introduces two low-rank matrices for the input weight matrix and the output weight matrix. The following is a detailed description of the LoRA logic.
### LoRA introduction
#### 1. initialization and pre-training.
- LoRA first assumes that there is a large-scale model that has been pre-trained and that these models have been trained on large-scale generalized datasets to learn rich language representations.
#### 2. low-rank matrix introduction:
- In the fine-tuning phase, LoRA does not directly update all the parameters. Instead, the method introduces two low-rank matrices.
#### 3. parameter update strategy:
- For each linear layer in the original model, LoRA computes the fine-tuned weight matrix using the following mathematical formula:

W' = W + U @ V^T

where W is the weight matrix of the original model, W' is the fine-tuned weight matrix, and U and V are the low-rank matrices associated with that linear layer, respectively.

#### 4. fixing and updating:
- During the fine-tuning process, LoRA fixes all parameters W of the original model and optimizes only the low-rank matrices U and V.
- This strategy allows the model to adapt to task-specific data by adjusting the low-rank matrices while maintaining most of the pre-training knowledge.
#### 5. local adaptation and global impact:
- The low-rank matrix is used to change the global impact of the original weight matrix through the above equation.
#### 6. resource efficiency and fast adaptation
- Since only two low-rank matrices need to be updated, LoRA significantly reduces the computational resource requirements and time costs during the fine-tuning of data parameters.
- In addition, due to its high parameter efficiency, LoRA is able to converge faster and adapt to task-specific data, and occasionally achieves good performance even with limited computational resources.
#### 7. Parallelization capability
- LoRA is updated in a way that is independent of the structure of the original model, which allows it to improve the speed of fine-tuning in parallel computation.

Overall, the logic of LoRA lies in the efficient updating of the parameters of a large pre-trained model through the use of two low-rank matrices, while also maintaining the original structure of the model and most of the training knowledge. The advantage of this approach is the ability to quickly adapt to new tasks with limited resources and maintain a high level of performance.

<img src="LoRA method.png" alt="LoRA method">

# Alpaca-LoRA model
Alpaca-LoRA is a large-scale language model optimization technique that combines the Alpaca model and the LoRA (Low-Rank Adaptation) method.
### The following is a detailed description of Alpaca-LoRA:
#### 1. Alpaca model:
- Alpaca is a large-scale language model designed to provide high-quality language comprehension and generation capabilities.
- Alpaca is based on the transformer architecture introduced in the previous section, and learns a rich language representation from a large amount of training data.
#### 2. application of LoRA method:
- In Alpaca-LoRA, the LoRA method is used to fine-tune the pre-trained Alpaca model to the specific domains needed for the research.
- LoRA uses two low-rank matrices to fine-tune against the parameters in the model.
#### 3. parameter efficiency and fast adaptation:
- The efficiency and speed of the fine-tuning process can be improved by using this model to significantly reduce the number of parameters needed for updating.
- This parameter efficiency allows the model to adapt quickly to new tasks and maintain a high level of performance with limited computational resources.
#### 4. Retention of pre-training knowledge:
- The present model fixes the original parameters of the Alpaca model in the fine-tuning and optimizes only the low-rank matrices U and V.
- This strategy helps to retain most of the linguistic knowledge learned by the Alpaca model in the pre-training phase, while allowing the model to adapt to new task requirements.
#### 5. parallelization and scalability:
- Since LoRA is updated in a way that is independent of the structure of the original model, the present model has good parallelization capabilities and can be further fine-tuned for speed.
- This feature makes the present model highly scalable in large-scale deployments and distributed training environments.

In summary, the model is fine-tuned using the LoRA approach to the transformer-based Alpaca model so that the Aplaca model can be applied to a variety of natural language processing tasks, including text categorization, sentiment analysis, question and answer systems, machine translation, and text generation.

# Data process
- Step 1: First load pandas and then read the file to be processed into df. Then calculate the Average sentiment score corresponding to each AuthorID and store the result in df1.

<img src="load_data.jpg" alt="load_data">

- Step 2: First create two lists has_read and Not_available. Then iterate through the AuthorID in df, if the ID is not_available then skip this loop, otherwise check whether the ID has been read, if it has been read then continue, otherwise add this author to the has_read list. Then start calculating the sentiment score for each reader, then store the sentiment score in the sentiment[] array.

In short, this step iterates through the entire DataFrame with an outer loop, and then determines whether the average sentiment score of each record belongs to the positive, neutral or negative sentiment with an inner loop.
- Step 3: Store these sentiment labels in final_sentiment and insert them into the corresponding positions of the sentiment[] array.

<img src="step_2.jpg" alt="step_2">

- Step 4: Import the random library and set the random number seed to 42. initialize two empty lists dataset_data and has_read. then calculate how many different author_id's there are.
- Step 5: Filter out the author_id's that are not in the Not_available list. for each remaining id, check if it appears in the has_read list, if it does pass it directly, otherwise add it to has_read list.
- Step 6: Store the sentiment output in data_dict before appending it to dataset_data list. Finally store the above output as .json file.

<img src="step_3.jpg" alt="step_3">

# Data Analysis
- Step 1: Import the nvidia-smi module, create a new virtualenv, install huggingface-hub, transformers and torch.
<img src="da1.jpg" alt="da1">

- Step 2: Initialize a language model object using the pre-trained model LlamaForCausalLM from Hugging Face and run it with floating-point 16-precision to improve performance.
<img src="da2.jpg" alt="da2">

- Step 3: Clone the online repository locally and read the dataset from the specified file using the load_dataset function from the HuggingFace library. Finally, the prompt method is generated to enter a piece of text describing the task and provide contextual information to get a response from the robot.
<img src="da3.jpg" alt="da3">

- Step 4: This paragraph has two main functions, tokenize and generate_and_tokenize_prompt.

#### Explanation for functions
Among them, the tokenize function is used to convert the prompt to a list of token IDs. It accepts two arguments: the prompt string and a boolean indicating whether to add an EOS token to the end of the prompt. If there is no EOS token, it checks whether results['input_ids'][-1] is equal to tokenizer.eos_token_id. If it is, it appends the EOS token to results['input_ids'] and updates attention_mask and label accordingly. otherwise, it directly copies results['input_ids'][-1] to results['input_ids']. 'input_ids'] as new 'labels'. Finally return the processed results.

The generate_and_tokenize_prompt function takes a data point, calls the generate_prompt method to generate the complete prompt, and then calls the tokenize method to encode it. The encoded complete prompt is returned.
<img src="da4.jpg" alt="da4">

- Step 5: Configure some key parameters of the LoRA algorithm and prepare the model to be used for micro-stepping training.

#### The role of each parameter is broadly described here:
LORA_R (integer): this parameter determines the size of the Lora architecture.

LORA_ALPHA (real number): This parameter controls the regularization strength.

LORA_DROPOUT (real): This parameter determines the ratio of the dropout.

LORA_TARGET_MODULES: It is a list containing the target modules.

BATCH_SIZE (integer) and related parameters: These parameters determine the batch size, batch size within a batch, and other related training details.
<img src="da5.jpg" alt="da5">

- Step 6: Micro-stepping the pre-trained model using the TRANSFORMER library.

#### First define the training parameters, where:
per_device_train_batch_size: training batch size on each device.

gradient_accumulation_steps: step size for gradient accumulation.

Next initialize the DataCollatorForSeqSeq object, which is responsible for feedforward delivery.

- Step 7: Create a Trainer instance, which consists of the following components.

#### Explanation for factors
model: The PyTorch model responsible for performing predictions and other computations.

train_dataset: The training dataset.

eval_dataset: Evaluation/test dataset.

args: Dictionary containing various hyperparameters.

data_collator: A data collector responsible for combining multiple samples into a single tensor.
<img src="da6.jpg" alt="da6">

- Step 8: Train the model.

- Step 9: Import the relevant dependency packages for TensorBoard and HuggingFaceHub, and then load the Tensorboard log file. Then refresh the Tensorboard and submit the model to the Hugging Face Hub account.
<img src="da7.jpg" alt="da7">

# Limitation:
The model in this study is not able to make quantified predictions, and is only able to make POSITIVE, NEGATIVE, and NEUTRAL predictions. The limitation of this model is that it is not able to output the exact trend and values, which can be improved in the future research.

# Future research:
As of now, the parts of the model that can be improved in this study include the following:
- Larger dataset, the dataset of this study can be enlarged by collecting data from more forums and authoritative websites that are updated in real time.
- More parameters, the final results can be made more accurate and comprehensive by introducing more evaluation parameters.
- More computer resources, better performance gpu can be used to train the model and finally achieve accurate numerical prediction.

```
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{henighan2020scaling,
  title={Scaling laws for autoregressive generative modeling},
  author={Henighan, Tom and Kaplan, Jared and Katz, Mor and Chen, Mark and Hesse, Christopher and Jackson, Jacob and Jun, Heewoo and Brown, Tom B and Dhariwal, Prafulla and Gray, Scott and others},
  journal={arXiv preprint arXiv:2010.14701},
  year={2020}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@inproceedings{rush2018annotated,
  title={The annotated transformer},
  author={Rush, Alexander M},
  booktitle={Proceedings of workshop for NLP open source software (NLP-OSS)},
  pages={52--60},
  year={2018}
}
```
