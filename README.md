# Social Media Sentiment Analysis Using a Fine-tuned Large Language Model
- This project is an initial preparation phase for social media sentiment analysis using a fine-tuned large language model. There are certain shortcomings, which will be addressed and improved in the future.
## Project Information:
- Author (in no particular order): Yiwei Liang; Jiaolun Zhou.
- Instructor: Prof.Luyao Zhang
- Project Summary: At this stage, the project uses the Llama2 , LoRA method and the Alpaca LoRA model for the study, based on user comments collected from online forums to train a model capable of making an analysis of cryptal price data based on daily forum data. The model is not able to analyze the cryptal price quantitatively, but only able to make "positive", "negative" and "neutral" judgments on the cryptal price trend. In the future, larger orders of magnitude of data and Llama30b can be used to correct this limitation.

Upon completion of this project, it is expected to be put into practical production in the following areas:
- Real-time market insights: By analyzing forum data, our model can provide implementation insights into the virtual currency market, including investor sentiment, the impact of news events, and possible price fluctuations.
- Risk Management and Decision Support: Financial institutions, investors and traders can utilize our model to assess investment risk, specify trading strategies and make more accurate investment decisions.
- Early Warning System: Our model can be developed into an early warning system that identifies market anomalies and potential risk events in advance, helping market participants, policy makers, and academics to adjust their strategies in a timely manner.
- Educational and Research Tools: Our model can be used as an educational and research tool to help students, academics and researchers better understand the dynamics and influences of the virtual currency market.
- Industry standard and innovation: Our model can become the new standard for virtual currency price prediction and drive technological innovation and application in related fields.
- Extension to other financial markets: Our model can be extended to other financial markets. Successful virtual currency price prediction methods can help make accurate predictions for stocks, bonds, commodities, and other areas, providing valuable assistance to the broader financial sector.
Overall, our project has great potential for the industrial sector, and at the same time, our project is innovative, and at this stage, we are not able to search Google Scholar for studies related to putting the Alpaca LoRA model into financial analysis and forecasting.

# Comparison of the performance of virtual machines of three platforms AWS, Azure, Google Cloud in machine learning:
This paragraph is used to elucidate the strengths and weaknesses of each of the three platforms of AWS, Azure, Google Cloud for VMs in machine learning and to provide recommendations for platform selection for this and future research.
## AWS (Amazon Web Services) Advantages:
- EC2 Instance Diversity: the AWS EC2 provides several instance types optimized specifically for machine learning, such as the P3/P4d series, designed specifically for GPU-accelerated deep learning training. Meanwhile, EC2 is equipped with the latest NVIDIA Tesla V100 or A100 GPUs; in addition to inference-optimized Inf1 instances with custom ASIC chips, Amazon Inferentia.
- Hosted Service: AWS SageMaker is a fully hosted service that allows users to easily build, train and deploy machine learning models on a variety of pre-built or customized EC2 instances, a design that simplifies the entire ML lifecycle management.
- Integration Services: AWS Glue, S3, Redshift, and other services seamlessly integrate with EC2 to facilitate data preparation and ETL operations, as well as large-scale data storage and analysis.
- Extensive ecosystem: AWS Marketplace offers a rich set of third-party machine learning tools and pre-trained models, making it easy to quickly build and scale solutions.
## AWS (Amazon Web Services) Disadvantages:
- Cost: While AWS offers a flexible pay-as-you-go model, the cost of certain advanced machine learning instances can be relatively high, especially for large-scale training tasks that run for long periods of time.
- Complexity: With so many services, users need to spend time finding the best combination of resources. In addition, as ML workloads change and resources need to be dynamically adjusted, more O&M experience may be required to optimize cost and performance.
- Data transfer costs: Although AWS provides a robust global infrastructure, migrating large amounts of data from local environments to S3 or other storage services can incur high outbound data costs. The cost of data movement is a consideration for machine learning projects that rely on large-scale datasets for training.
## Microsoft Azure Advantages:
- High-performance VMs: Azure provides NCv3/NCv4 series VMs with built-in NVIDIA Tesla V100 or A100 GPUs suitable for deep learning training. In addition, Azure Machine Learning Compute can be dynamically scaled to fit training needs.
- Azure Machine Learning Service: Similar to AWS SageMaker, Azure ML provides a one-stop machine learning development environment, including automated model training, versioning and deployment capabilities.
- Integration and Collaboration: Azure Synapse Analytics, Data Factory and other data services are tightly integrated to help streamline data pipelines and machine learning processes. Microsoft works closely with the open source community and has good support for frameworks such as PyTorch and TensorFlow.
- Enterprise-level compatibility: for organizations already using the Microsoft technology stack, Azure tends to offer smoother migration paths and integration options.
## Microsoft Azure Disadvantages:
- Low pricing transparency: Azure's service pricing model is not intuitive enough, especially when it comes to scaling large-scale compute resources on demand. Understanding and optimizing costs can be a challenge for long-running machine learning training tasks.
- Compatibility and migration issues: For customers using non-Microsoft ecosystems, deploying and maintaining machine learning models built on open source frameworks such as TensorFlow on Azure may face compatibility or migration difficulties compared to other platforms.
## Google Cloud Platform (GCP) Advantages:
- High-performance computing: GCP provides instances of NVIDIA T4, V100, or A100 GPUs in Compute Engine for efficient machine learning training.Cloud TPUs are tensor processing units designed specifically for machine learning, and are particularly adept at accelerating training in the TensorFlow framework.
- Deep Integration: Google Cloud AI Platform provides a complete solution from data preparation to model training, evaluation, and deployment, and is deeply integrated with interactive development environments such as Google Colab.
- Open Source Contributions and Optimization: As a major contributor to TensorFlow, GCP has an advantage in native support and optimization, making the experience better for developers using TensorFlow.
- Innovative capabilities: Google continues to invest in cutting-edge AI research and rapidly translates these results into cloud services such as the AutoML product, which enables non-experts to build high-quality machine learning models.
## Google Cloud Platform (GCP) Disadvantages:
- Service lock-in risk: for users who rely on Google's own products, if they choose to leave GCP and migrate to other platforms there may be certain service migration and technology adaptation issues.
## Summary:
All three platforms have the advantages of each deep integration, however, users need to calculate carefully when it comes to price, which is not a low overhead. In addition to this, try not to change the platform you are using, as interoperability between platforms is poor.
## Table Of Contents:
1. [**Self Introduction**](./Author)
2. [**Research Questions**](./First_train/Research_questions)
      - [Research questions](./First_train/Research_questions/#Researchquestions)
      - [Significance](./First_train/Research_questions/#Significance)
      - [Study 1: "Using GPT-3 for Stock Market Prediction"](./First_train/Research_questions)
      - [Study 2: "Cryptocurrency price prediction using traditional statistical and machine-learning techniques: A survey"](./First_train/Research_questions)
      - [Study 3: "Forecasting the price of Bitcoin using deep learning"](./First_train/Research_questions)
3. [**Methodology**](./First_train/Methodology)
   - [Introduction to Methodology](./First_train/Methodology/#Thisresearchusesthefollowingsteps:)
   - [Llama2 model](./First_train/Methodology/#Llama2_model)
   - [Transformer architecturel](./First_train/Methodology)
   - [LoRA method](./First_train/Methodology/#LoRA_method)
   - [Alpaca-LoRA model](./First_train/Methodology/#Alpaca-LoRA_model)
   - [Data process](./First_train/Methodology/#Data_process)
   - [Data Analysis](./First_train/Methodology/#Data_Analysis)
   - [Limitation](./First_train/Methodology/#Limitation:)
   - [Future research](./First_train/Methodology/#Future_research:)
4. [**Application Scenario**](./First_train/Application_Scenario)
5. [**Data**](./First_train/data)
6. [**Code**](./First_train/code)
7. [**Result**](./First_train/Result)


## More About the Authors
### Yiwei Liang
- <img src="Author/Yiwei.jpg" alt="Yiwei" width="220"/>
#### Self-introduction:
I am Yiwei Liang.
### Jiaolun Zhou
- <img src="Author/Jiaolun.jpg" alt="Yiwei" width="220"/>
#### Self-introduction:
I am Jiaolun Zhou from Computer Science at Duke Kunshan University, working on IoT and machine learning, and I am honored to be able to participate in this research and contribute to virtual currency price prediction and the future development of the financial market.
