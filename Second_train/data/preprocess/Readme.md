# Introduction:
This research will be divided into several steps to process the data and train the model, and this page explains the principles and advantages of the techniques used in the research process.

#### Below are the three sections that will be highlighted on this page:
- 1. Data processing + NLP + Polynomial Regression
- 2. Big data packages Packages for processing big data
- 3. Parallel Computing + Parallel Computing
#### Data processing + NLP + Polynomial Regression
**MATHEMATICAL PRINCIPLES**: polynomial regression is a nonlinear regression model that establishes a nonlinear relationship between the dependent and independent variables by mapping the independent variable to a higher power.

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

In this equation m is the sample size, and by taking the partial derivative of RSS and making it equal to zero, a set of linear equations can be obtained for the \beta parameter, and solving this set of equations yields a least squares estimate of \beta.
- Gradient descent: Gradient descent is an optimization algorithm used to find the local minima of the loss function. For polynomial regression, we can define a loss function such as the mean square error, then compute the gradient of that function with respect to each coefficient and update the parameter values in the direction opposite to the gradient until a predefined stopping condition is reached (e.g., the gradient is sufficiently small or the number of iterations reaches an upper limit).
- Regular equations: for linear regression problems, when the dimensionality of the independent variables is not very high, the parameters can be solved directly by matrix operations. In polynomial regression, even though the model is nonlinear, the model can be transformed into linear regression form during processing, so in regular use, researchers usually adopt the method of formal equations to solve the optimal parameters at one time, the equations are as follows:

$$ X^T X \boldsymbol{\beta} = X^T y $$

