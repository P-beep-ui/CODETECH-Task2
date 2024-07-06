Name:BHUPATHI VENKATA SAI PRAMOD KUMAR
Company:CODETECH IT SOLUTIONS
ID:CT08DS3654
Duration:July 1st to August 1st

Overview of the Project

Project:IMDB movies POSITIVE or NEGTIVE
![Screenshot (2)](https://github.com/P-beep-ui/CODETECH-Task2/assets/174769475/58470470-e8b7-48da-92aa-d80d372a3a31)




Objective
The objective of this script is to build a machine learning model to predict the sentiment (positive or negative) of movie reviews using a logistic regression classifier.

Key Activities
Import Libraries:

Import necessary libraries including NumPy, Pandas, and scikit-learn modules.
Load Dataset:

Load the IMDb movie reviews dataset from a CSV file using Pandas.
Prepare the Data:

Extract features (movie reviews) and labels (sentiments).
Convert sentiment labels to binary values (1 for positive, 0 for negative).
Split the Dataset:

Split the dataset into training and testing sets.
Preprocess the Data:

Use TF-IDF Vectorizer to transform the text data into numerical features suitable for machine learning.
Build and Train the Model:

Create and train a logistic regression model using the training data.
Evaluate the Model:

Make predictions on the test set and evaluate the model’s accuracy.
Predict Sentiment:

Define a function to predict the sentiment of new movie reviews using the trained model.
Technology Used
Libraries:

NumPy: For numerical operations.
Pandas: For data manipulation and analysis.
Scikit-learn: For machine learning algorithms and tools.
train_test_split: To split the dataset.
TfidfVectorizer: To preprocess text data.
LogisticRegression: To build and train the model.
accuracy_score: To evaluate the model's performance.
Dataset:

IMDb movie reviews dataset in CSV format.
Key Insights
Text Preprocessing:

Using TF-IDF Vectorizer helps in converting textual data into numerical features by considering the importance of words in the documents, which improves the model's performance.
Model Selection:

Logistic Regression is a suitable choice for binary classification tasks like sentiment analysis, providing interpretable results and decent performance.
Evaluation:

The model achieved a certain level of accuracy on the test set, indicating its effectiveness in predicting sentiments.
Practical Application:

The model can be used to automatically predict the sentiment of new movie reviews, demonstrating the practical application of machine learning in natural language processing tasks.
Reproducibility:

Setting a random state in train_test_split ensures that the results are reproducible.
