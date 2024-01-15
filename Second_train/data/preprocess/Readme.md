# Introduction:
This research will be divided into several steps to process the data and train the model, and this page explains the principles and advantages of the techniques used in the research process.

#### Below are the three sections that will be highlighted on this page:
- 1. [**Data processing + NLP + Polynomial Regression**](./Second_train/data/preprocess/Data_processing_+_NLP_+_Polynomial_Regression)
- 2. [**Big data packages Packages for processing big data**](./Second_train/data/preprocess/Comparison_for_various_types_of_big_data_processing_packages)
- 3. [**Parallel Computing**](./Second_train/data/preprocess/Parallel_Computing)
#### Data processing + NLP + Polynomial Regression
**Mathematical Principles**: polynomial regression is a nonlinear regression model that establishes a nonlinear relationship between the dependent and independent variables by mapping the independent variable to a higher power.

The general form of a polynomial regression model can be written:

 $$ \[ y = f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_n x^n + \epsilon \] $$

Among them:
- $\( y \)$ is the dependent variable (response variable).
- $\( x \)$ is the independent variable (characterization variable), which is raised to different powers in polynomial regression to capture nonlinear relationships.
- $\( \beta_0, \beta_1, ... , \beta_n \)$ are the coefficients of the model, where $\( \beta_0 \)$ is the intercept term, $\( \beta_1, ... , \beta_n \)$ are the coefficients of the model, where $\( \beta_0\)$ is the intercept term, and $\( \beta_1, ... \beta_n \)$ corresponds to the coefficients of polynomials of order $\( x \) up to order \( n \)$.
- The $\( \epsilon \)$ is a random error term, which is usually assumed to be independently and identically distributed noise with zero mean.

In practice, we estimate the model parameters $\(\beta \)$ by minimizing the residual sum of squares (RSS) or by using regular equations, for example. Once the optimal coefficient estimates are obtained, the model can be utilized for predictive analysis.
 
**Mathematical Foundations**:
- Least Squares Estimation: in polynomial regression, the most commonly used method of parameter estimation is the least squares method. The goal of this method is to minimize the sum of squares of the residuals, which is the sum of squares of the difference between the predicted and true values for all sample points. The mathematical representation is:

$$ RSS(\beta) = \sum_{i=1}^{m}(y_i - f(x_i))^2 $$

In this equation m is the sample size, and by taking the partial derivative of RSS and making it equal to zero, a set of linear equations can be obtained for the $\beta$ parameter, and solving this set of equations yields a least squares estimate of $\beta$.
- Gradient descent: Gradient descent is an optimization algorithm used to find the local minima of the loss function. For polynomial regression, we can define a loss function such as the mean square error, then compute the gradient of that function with respect to each coefficient and update the parameter values in the direction opposite to the gradient until a predefined stopping condition is reached (e.g., the gradient is sufficiently small or the number of iterations reaches an upper limit).
- Regular equations: for linear regression problems, when the dimensionality of the independent variables is not very high, the parameters can be solved directly by matrix operations. In polynomial regression, even though the model is nonlinear, the model can be transformed into linear regression form during processing, so in regular use, researchers usually adopt the method of formal equations to solve the optimal parameters at one time, the equations are as follows:

$$ X^T X \boldsymbol{\beta} = X^T y $$

It is important to note that when using polynomial regression, one needs to be wary of overfitting, especially when choosing higher order polynomials. Model complexity can be controlled and generalization improved through cross-validation, regularization (e.g., ridge regression or Lasso regression), or other model selection techniques.
#### Advantages of using polynomial regression:
- Ability to fit non-linearly: Polynomial regression is able to deal with complex linear trends published in datasets, which is particularly useful in predicting the dynamics of financial markets, where market behavior is often influenced by a variety of factors and where there may be non-linear mutual exclusionary effects between these influences.
- Simple and intuitive: Compared to other complex nonlinear models, polynomial regression models have a relatively simple structure that is easy to understand and interpret. This saves time in the initial use of this study and can be used to check whether the data meets the requirements for model training.
- Statistical Inference: Due to its extension of the linear regression framework, polynomial regression also supports standardized methods for statistical significance testing and parameter estimation.
#### Use of polynomial regression in related fields:
- **Title** : Forecasting exchange rates with linear and nonlinear models
- **Content** : This study demonstrates the reference value of polynomial regression in financial market forecasting by comparing the different performances of linear and nonlinear models in terms of their analytical capabilities for financial market data generated by the models.
- **Improvement**: Compared to this study, this study uses real-time data from forums, and by using this method of data collection, this study is able to achieve more realistic forecasts.
#### Comparison for various types of big data processing packages:
- **Apache Hadoop**: Hadoop is one of the earliest and best-known big data processing frameworks, and its core components include HDFS for storing massive amounts of data and parallel computing through the MapReduce programming model.The Hadoop ecosystem is huge, and includes several companion projects, such as Hive (a data warehousing tool), Pig (a scripting language), HBase (columnar), and so on. HBase (columnar database) and many other companion programs for batch processing tasks. However, Hadoop MapReduce is relatively slow to execute because it requires disk IO operations between each stage; it is not optimal for the processing needs designed in this study for real-time or streaming data in forums.
- **Apache Spark**: The main advantage of Spark over Hadoop is speed, which uses in-memory computing technology to cache intermediate results in memory to speed up the process.Spark supports a variety of computational paradigms, including batch processing, interactive querying (Spark SQL, Hive on Spark), stream processing (Spark Streaming/S Structured Streaming) as well as machine learning (MLlib) and graphics computing (GraphX). This makes Spark a more generalized data processing platform. However, the biggest drawback of this package is the expense. While Spark running in memory can significantly improve performance, if the amount of data exceeds the available memory, it may require frequent data exchanges with disk, which can impact performance. In addition, for very simple batch tasks, Spark's overhead may be greater than MapReduce.
- **Apache Flink**: Flink is a framework for real-time and stream processing, but also efficiently handles batch tasks. It provides exact-once semantic guarantees, which are critical for data consistency in the financial sector.Flink supports event time windows and state management for complex event-driven scenarios and continuous computation. However, as a new package, this package is not rich in community resources, compared to some other more mature frameworks, Flink's learning curve may be steeper, community resources and maturity may be slightly inferior to Spark. but in some specific application scenarios, especially where real-time and fault-tolerance requirements are high, Flink's advantages are obvious.
- **Apache Storm**: Storm focuses on real-time stream processing, providing low-latency and high-throughput service guarantees to ensure that the data will not be lost and at least once (at-least-once) messaging guarantees.Storm is suitable for continuous real-time data streaming analytics and early warning systems. However, Storm is not as powerful as Flink for complex window processing and state management, and its API is relatively primitive and writing complex logic can be cumbersome.Storm does not directly support batch processing, and if there is a need to mix batch and stream processing in the study, additional technology stack integration may be required.
- **Microsoft Azure Data Factory**: Azure Data Factory is a cloud service that provides a one-stop ETL (Extract, Transform, Load) solution that makes it easy to integrate data from different sources. Combined with Azure HDInsight (which includes components such as Hadoop and Spark) and Azure Databricks, it allows for large-scale batch and streaming analytics, and Microsoft also provides services and APIs related to machine learning and NLP, such as the Azure Machine Learning Service.However, if the research is based entirely on a local environment or if you do not intend to use Azure cloud services, then Data Factory is not the best choice. In addition, this package is not open source and may have some limitations for specific custom algorithm development and tuning compared to open source frameworks.
- **Snowflake Computing**: Snowflake is a cloud-native data warehouse service that supports rapid querying of large amounts of structured data and the ability to seamlessly integrate multiple data sources. While it is primarily SQL-centric, by combining it with external tools such as Python, R, or Spark, it can fulfill data analysis and modeling needs, including processing of text data and machine learning applications. However, Snowflake itself does not directly provide complex NLP processing capabilities, and needs to be used in conjunction with other NLP libraries; and for real-time stream processing needs, it may need to additionally rely on Kafka or other stream processing systems.
- **Amazon Web Services (AWS) Stack**: AWS provides a full suite of big data solutions, including EMR (Elastic MapReduce, based on Hadoop and Spark), Kinesis (stream processing), Redshift (data warehouse), SageMaker (machine learning platform) and Comprehend (NLP service). These services are highly integrated with each other and can be directly applied to the entire process from data collection and preprocessing to model training and deployment. However, for research teams that want to run their big data projects on local servers or private clouds, the full benefits of AWS may only be fully realized in the cloud. Additionally, despite the abundance of AWS services, the cost of use will be higher compared to open source solutions.
- **Conclusion**: Based on this research for the processing needs of real-time data in forums, Apache Spark may be the most suitable choice because it not only has strong machine learning and NLP support, but also balances the needs of batch and stream processing. However, if an online deployment is considered, the big data processing packages offered by Azure and Aws are also possible choices because of their own performance.

#### Parallel Computing:
- Principle: Parallel computing is the process of accelerating computation by breaking down a large and complex problem into multiple small sub-problems that can be processed at the same time, and executing these sub-tasks simultaneously using multiple computers, multiple processor cores, or multiple threads within the same processor. At the hardware level, this may involve heterogeneous parallel computing architectures such as shared memory systems (e.g., multi-core CPUs), distributed memory systems (e.g., clusters or cloud computing environments), and GPUs. At the software level, the design of parallel algorithms typically includes task partitioning, communication policies, and synchronization mechanisms. For example, in MPI (Message Passing Interface), processes collaborate with each other through message passing, while in programming models such as OpenMP or CUDA, collaboration between threads is realized by means of shared memory.
- Mathematical Principles:

**Amdahl's law**: Amdahl's law describes the maximum speedup ratio that can be achieved by increasing the number of processors in the presence of serial sections. Assuming that the proportion of the part of the system that can be executed in parallel is f, then the remaining part is the part that must be executed serially, and its speedup ratio S can be expressed by the following equation: the speedup ratio $\( S \)$ is defined by Amdahl's law as follows: $$\( S = \frac{1}{(1 - f) + \frac{f}{p}} \)\)$$

**Gustafson-Barsis Law (also known as Gustafson's Law)** : The Gustafson-Barsis law is an extension of Amdahl's law that takes into account the situation where the size of the problem grows proportionally as hardware resources grow. In this case, the total workload W can be divided into a serial part $W\_S$ and a parallel part $W\_p$ , and the processing power of the whole system grows linearly as more processors are added. The basic idea is that as the number of processors grows, not only does the parallel part speed up, but the size of the problem being processed also increases.

- Strengths:

**Improved computational efficiency**: parallel computing significantly reduces overall computation time because multiple processors can work simultaneously, dramatically increasing the speed of operations.

**Solve Large-Scale Problems**: For large-scale data sets or highly complex computing tasks that cannot be processed in real-time by a single processor, such as big data analysis, simulation, machine learning training, etc., results can be obtained in a reasonable amount of time through parallel computing.

**Resource Utilization**: Through parallel computing, idle computing resources can be effectively utilized, reducing the risk of single point of failure and enhancing the fault tolerance and reliability of the system.

**Scalability**: With the development of technology, the computing power can be linearly expanded by adding more processors or computing nodes to adapt to the growing demand for data processing.

- Parallel computing is suitable for this study for the following reasons:

**Data preprocessing**: the NLP processing session requires word splitting, word vectorization, etc. for a large amount of text, and these operations are well suited for parallelization, where each core can process a portion of the document independently, speeding up the overall processing speed.

**Feature Extraction**: polynomial regression and other machine learning model construction will involve a large amount of feature engineering, many of these steps (e.g., statistical metrics computation, matrix operations, etc.) have inherent parallelism.

**Model Training**: During deep learning and machine learning model training, gradient computation and parameter updates can be performed using data-parallel, model-parallel, or mixed-parallel strategies to accelerate training convergence using multi-core CPUs or GPUs.

**Streaming Data Processing**: If the research also includes real-time or near-real-time analysis of news data streams, the use of parallel stream processing frameworks (such as Apache Flink or Spark Streaming) enables the processing of newly generated data in real-time and rapid prediction.

**Conclusion**: Parallel computing enables research teams to process large amounts of text in a shorter period of time and efficiently build and optimize models, thus improving the ability to predict financial data.
```
@book{padua2011encyclopedia,
  title={Encyclopedia of parallel computing},
  author={Padua, David},
  year={2011},
  publisher={Springer Science \& Business Media}
}
@book{quinn1994parallel,
  title={Parallel computing theory and practice},
  author={Quinn, Michael J},
  year={1994},
  publisher={McGraw-Hill, Inc.}
}
@article{bissoondeeal2008forecasting,
  title={Forecasting exchange rates with linear and nonlinear models},
  author={Bissoondeeal, Rakesh K and Binner, Jane M and Bhuruth, Muddun and Gazely, Alicia and Mootanah, Veemadevi P},
  journal={Global Business and Economics Review},
  volume={10},
  number={4},
  pages={414--429},
  year={2008},
  publisher={Inderscience Publishers}
}
@article{heiberger2009polynomial,
  title={Polynomial regression},
  author={Heiberger, Richard M and Neuwirth, Erich and Heiberger, Richard M and Neuwirth, Erich},
  journal={R Through Excel: A Spreadsheet Interface for Statistics, Data Analysis, and Graphics},
  pages={269--284},
  year={2009},
  publisher={Springer}
}
```
