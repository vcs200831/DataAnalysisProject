# DataAnalysisProject

This project focuses on performing an exploratory data analysis (EDA) on the Titanic dataset. The dataset contains information about the passengers aboard the Titanic, including their demographics, ticket class, survival status, and more.

## Project Overview
The project aims to gain insights into the factors that influenced the survival of passengers on the Titanic. It involves several steps, including data cleaning, analysis, and visualization. The main goals of the project are as follows:
1. Exploratory Data Analysis (EDA): Gain an understanding of the dataset by examining its structure and contents. Display the first few rows of the dataset and calculate summary statistics to get an overview of the data. Check for missing values and perform necessary data cleaning steps.
2. Survival Analysis: Investigate the survival rate based on different variables such as gender and passenger class. Calculate the survival rate by gender and passenger class using group by operations. Visualize the results using bar charts and pie charts. 
3. Additional Functionality: Extend the analysis by adding more features. In this project, we explore the age distribution of survivors and non-survivors using histograms. Additionally, we create a scatter plot to analyze the relationship between age and fare, colored by survival status.
4. Model Building: Build a logistic regression model to predict the survival status of passengers based on selected features. Split the dataset into training and testing sets, preprocess the data, and train the logistic regression model. Evaluate the model's accuracy on the testing set.

## Dependencies
To run the project, you need to have the following dependencies installed:
- pandas
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Clone the project repository to your local machine.
2. Download the Titanic dataset (`train.csv`) from [source link](https://www.kaggle.com/c/titanic/data).
3. Place the downloaded dataset file in the `datasets` folder within the project directory.
4. Open a terminal or command prompt and navigate to the project directory.
5. Run the `main.py` file using the Python interpreter.

## Conclusion and Future Work
The Titanic Data Analysis project provides valuable insights into the factors affecting passenger survival on the Titanic. Through exploratory data analysis, we have uncovered trends and relationships between variables such as gender, passenger class, age, fare, and survival.
In future iterations of the project, we can consider expanding the analysis by exploring additional variables or incorporating more advanced machine learning algorithms for prediction. Furthermore, we can enhance the visualizations by adding interactive features or creating a web-based dashboard to make the analysis more accessible and user-friendly.
Overall, this project showcases my data analysis skills and ability to derive meaningful insights from real-world datasets. It demonstrates my proficiency in data preprocessing, exploratory data analysis, visualization, and basic machine learning techniques.
