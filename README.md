# Healthcare-Insights

Decoding Health Habits: A Beginner's Walkthrough of a Data Science Project

Welcome to the world of data science! This document is designed to demystify a real-world project, breaking it down step-by-step. We'll explore how data scientists use measurable body signals—like blood pressure and cholesterol—and a set of powerful tools called machine learning to understand complex health habits, such as smoking and drinking.


--------------------------------------------------------------------------------


## 1. The Starting Point: A Common Healthcare Challenge

In healthcare, getting a complete and accurate picture of a patient's history is crucial for effective treatment. However, when asked about habits like smoking or drinking, patients may hesitate to share the full story. This could be due to a fear of judgment or other personal reasons. This lack of information creates a significant challenge for doctors, who need this context to make the best possible treatment decisions.

This project's goal is to see if machine learning can use measurable body signals to accurately infer these habits without relying solely on patient self-reporting. While the project's methods apply to both, the original analysis placed a primary focus on predicting smoking habits.

To tackle this problem, the project starts with the most important ingredient: data.


--------------------------------------------------------------------------------


## 2. The Raw Material: Getting to Know the Dataset

The analysis uses an anonymous public dataset called the "Smoking and Drinking Dataset." Originally collected by the National Health Insurance Service in Korea, it was made available for research on the platform Kaggle. This large dataset contains information from nearly one million individuals, with 991,320 rows and 24 columns of different "body signals" for each person.

While many metrics were available, the analysis found that a few key data points were especially impactful for understanding health habits. Here are five of the most important ones:

Health Metric	What It Tells Us (In Simple Terms)
SBP / DBP	Systolic and Diastolic Blood Pressure are the two numbers in a blood pressure reading. They measure the pressure in your arteries when your heart beats and when it rests.
Hemoglobin	is an iron-rich protein in red blood cells that is responsible for carrying oxygen from the lungs to the rest of the body.
SGOT_AST / ALT: These are liver enzymes. Elevated levels of Aspartate Transaminase (AST) and Alanine Transaminase (ALT) can indicate liver stress or damage.
HDL Cholesterol, High-Density Lipoprotein, is often called "good" cholesterol. It helps remove other forms of cholesterol from your bloodstream.

With this raw data in hand, the first essential step is to make sure it's clean and ready for analysis.


--------------------------------------------------------------------------------


## 3. Preparing for Discovery: The Data Cleaning Process

Before any analysis can begin, a data scientist must act like a quality inspector. This step, known as data cleaning, is critical for ensuring that the project's results are accurate and reliable. You wouldn't want to build a house on a shaky foundation, and similarly, you don't want to build a predictive model on messy data.

In this project, two main cleaning actions were performed:

1. Checking for Duplicates: The dataset was scanned for any identical, repeated entries. Duplicates were found and removed to prevent the same information from being counted multiple times, which could skew the results.
2. Checking for Missing Information: The team checked if any data points were missing (often called "null" values). Fortunately, the dataset was complete, and no missing values were found.

Now that the data is clean and trustworthy, it's time to start exploring it for initial patterns.


--------------------------------------------------------------------------------


## 4. Uncovering Clues: Key Insights from the Data

After cleaning, the next phase is Exploratory Data Analysis (EDA). Think of this as the detective work of data science, where you sift through the evidence to find initial clues and relationships.

First, the analysis revealed that some data points—like height, weight, and eyesight—were not very useful for this specific problem. Their values were generally within a normal range across all groups (smokers, non-smokers, drinkers, etc.) and didn't offer much predictive power, so they were disregarded for the final model.

The investigation did, however, uncover several significant insights:

* Insight 1: The Link Between Drinking and Cholesterol. The data showed that people who drink are more likely to have a wider range of cholesterol levels. While most people had normal levels, the individuals with exceptionally high cholesterol were more often found in the group that reported drinking habits, suggesting a connection between alcohol consumption and cholesterol variability.
* Insight 2: The Connection Between Smoking and Hemoglobin. A moderate positive correlation (0.45) was found between a person's smoking status and their hemoglobin levels. This suggests that as smoking habits increase (from never smoking to currently smoking), hemoglobin levels also tend to be higher.
* Insight 3: The Relationship Between Blood Pressure Readings. The analysis confirmed a well-known medical fact: systolic (SBP) and diastolic (DBP) blood pressure are strongly correlated (0.74). This means that when one number is high, the other tends to be high as well, which is a useful pattern for a model to learn.

These initial clues are invaluable, as they help guide the next and most powerful step: building predictive machine learning models.


--------------------------------------------------------------------------------


## 5. The Prediction Toolkit: Testing the Machine Learning Models

The core of this project is using machine learning to make predictions. The goal is to train a "model" on the cleaned data, allowing it to learn the complex patterns connecting a person's body signals to their smoking or drinking habits. Once trained, the model can then be used to predict the habits of a new person based only on their health metrics.

To find the best tool for the job, the team tested several different types of machine learning models. The table below shows how well each one performed at predicting drinking and smoking status.

***Machine Learning Model:*** A Simple Explanation, Drinking Prediction Accuracy, Smoking Prediction Accuracy

- ***Logistic Regression:*** A straightforward and effective model for predicting a "yes" or "no" outcome.	72.05%	67.50%

- ***Gaussian Naive Bayes***	is a model that uses probability to classify, assuming that each feature (like blood pressure) is independent of the others.	65.26%	64.98%

- ***Decision Tree***	works like a flowchart, asking a series of if-then questions about the data to arrive at a decision.	63.66%	61.48%

- ***Random Forest***	An "ensemble" model that combines many Decision Trees to make a more accurate and stable prediction.	72.38%	69.02%

- ***Gradient Boosting:*** A powerful "ensemble" model that builds a series of trees, where each new tree corrects the errors of the previous one.	72.83%	69.73%

- ***Artificial Neural Network (ANN):*** A model inspired by the human brain, excellent at finding very complex, non-linear patterns in large datasets.	73.24%	69.43%

Looking at these results, you might notice something interesting. The Artificial Neural Network (ANN) scored fractionally higher on drinking prediction (73.24%), while the Gradient Boosting model was the top performer for smoking prediction (69.73%) and was extremely competitive for drinking (72.83%). The project team selected the Gradient Boosting model for their final predictions, which teaches us a crucial lesson in practical data science: the "best" model isn't always the one with the single highest score on one task.

In real-world projects, data scientists often choose a model that performs consistently and reliably across all prediction goals. Given that Gradient Boosting was the top performer on one task and a close second on the other, it proved to be the most balanced and robust choice for the overall project. Factors like easier interpretability or faster training times can also make a consistently strong model more appealing than one that's slightly better on a single metric.

But prediction is just one part of data science. The team also used other techniques to find deeper patterns in the data.


--------------------------------------------------------------------------------


## 6. Finding Deeper Patterns: Clustering and Association Rules

Beyond predicting a single outcome, data science techniques can also uncover hidden groups and subtle relationships within a dataset. This project explored two such methods: clustering and association rule mining.

Grouping Similar Patients with Clustering

Clustering is a technique used to automatically group data points—in this case, patients—that share similar characteristics. The analysis identified three distinct patient groups within the dataset, each with a unique health profile:

***1. Cluster 0 (Kidney Health Focus):*** This group consisted of individuals whose body signals showed signs of potential kidney-related health issues.

***2. Cluster 1 (Healthier Profile):*** This group included patients with generally healthier body signals across the board compared to the other clusters.

***3. Cluster 2 (Taller, Higher Body Mass):*** This group was characterized by taller individuals who, like Cluster 0, also showed potential kidney-related health concerns.

Linking Liver Enzymes to Drinking Habits

Another technique, called Association Rule mining, was used to find "if-then" relationships between different variables. The analysis searched for connections between lab results and drinking habits.

The key finding was a mild association, suggesting that specific levels of the liver enzyme SGOT_AST are more commonly observed in individuals who do not drink. While not a strong predictor on its own, this finding points to a subtle pattern connecting liver function to alcohol consumption.

These findings, combined with the predictive models, lead to the project's conclusions.


--------------------------------------------------------------------------------


## 7. The Final Verdict: What This Means for Healthcare

This project successfully demonstrated that it is possible to use data science and machine learning to infer patient habits that are often difficult for doctors to obtain directly. By analyzing objective body signals, the models could predict smoking and drinking status with a reasonable degree of accuracy.

The most important real-world implication of this work is its potential to empower healthcare providers. Having a more complete and objective picture of a patient's habits—especially when self-reporting is incomplete—could help doctors make better, more informed treatment decisions. This, in turn, can lead to improved patient care and better long-term health outcomes.

Ultimately, this project is a perfect example of how data science can be applied to solve meaningful, real-world problems in healthcare.
