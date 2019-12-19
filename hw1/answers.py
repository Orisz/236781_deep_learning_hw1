r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
** Best K was k=3, but for some code runs it was k=1. 
Increasing k did not improve generalization very much for unseen data. 
Lets think about k, If we demand a huge k, so at the limit we are considering the entire train set.
Of course that is not a good practice since each time the label picked will be of the majority in the data set.
I.e. we will embrace the bias of the data set.
For too small k we are overfitting since k=1 means no generalization. In the case of mnist it seems that it 
doesn't effect much on the results.
The validation acc seems to drop around k=5. We noticed that we get ~91.8-92% test-accuracy for all k=1,2,3,4,5,6,7,8,9. **

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
** 
The point of using CV is to avoid overfitting the data.
2) The second notion of picking the model with the highest test accuracy is strictly overfiting.
The test set sole purpose is to evaluate the results of the model,
 and if we trained a good generalized model it will preform well on the test set.

1) The first notion will also harm our models generalization. if we are trainig a single model the notion might be fine.
 But since we are looking to set some hyper parameters we are looking (by splitting the data),
a model that will generalize the problem, and by extension will preform well for all the folds.
Picking the model that will preform the best on the entire train-set,
means we are over fitting our hyper parameters to the train set instead of trying to generalize them.
Over all, picking a model based on the entire train-set accuracy will preform not so well on the test set. **


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**The role of delta regarding the SVM hinge loss is to make sure that we choose the correct lable,
 by at least 'delta' distance from all other labels.
 And so delta has to be positive, so we choose the correct label, but it does not matter 'how positive' delta will be.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**1) The linear classifier is trying to distinguish between labels in a linear way by finding a linear hyper plain
 which separates between the classes.
The weights we learned are the best for the separation within delta between the classes.
We can also see that the weights we got for 'visualy similar' digits (E.G. '0' and '9' ) are close to each other.
It seems that the classification errors occurs when some example are really close to another digit.
 Close in the sense they 'look alike' another digit, and may also in the sense of the distance to the hyper plain.

 2) KNN looks for the 'closest' example/s to our new input in order to classify. 
 It always checks the entire training set with respect to the input.
 SVM on the other hand, tries to differentiate between whole classes.
  So essentially when trying to classify a new example we don't check all the training data, we just consider the hyper plains.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
** 1) Lets consider the following: If the rate was too low the loss would not converge so well(4-5 epoches only!).
Further more if the rate was too high we would see spikes in the loss and so the rate is obviously not too high.
To conclude we can say the rate was good.
2) With respect to the test set acc: ("Test-set accuracy after training: 87.0%") we can deduce that we are silghtly 
over fitted to the training set, since the acc results on the train-valid sets are better by ~8%.
Which was to be expected since there is a considerable gap between the validation and the training acc.
The fact that after epoches 4-5 we kept getting better results on the training set but didn't get any better on the valid
set shows us that the model is over fiting to the trainig set.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
"The ideal pattern would be if we see no error, meaning we will not see a big difference between y_hat and y.
That can be achieved if all the points would be along the y=0 line.
It is clear that using the CV gave us better results, as both the MSE improved, and the dots are closer
to the y=0 line. "

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**
'''
1) The function np.linspace gives us samples that are evenly spaces, while np.logspace gives us log spaced samples. 
  The reason we used log space is because that the loss is highly effected by the weights, more than it is 
  effected by lambda.
  And so, we wanted to make sure the changes we make large changes in the lambda, so we used log scale.
2) The number of times we trained the model is the number of variation in the parameters:
    #num_of_lambda_values * #num_of_degrees
    And so the total number is: 3*20 = 60 Times.
'''


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
