# Ferrer_Noble_MexEE402_CaseStudy

# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Ferrer, Wendell Glenn Don V.| 23-07657 | MEXE-4101 |
| Noble, Aedelbert D. | 23-00737 | MEXE-4101 |

## Notebook links

| Chapter | Ferrer | Noble |
|---|---|---|
| Ch1_2_3 | [link](https://colab.research.google.com/drive/1k3qXB9SRzeKiHNAI5hmcSgvg0ZBz5S2X?usp=sharing) | [link](https://colab.research.google.com/drive/1YQd-56bjY7qgkWE9a77sHb8Uk427NPQn?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1UPZSM9Or6eVB6WVcYavlxxX7zqW210U2?usp=sharing) | [link](https://colab.research.google.com/drive/11adtQlaX19OGzVgIkJAcjJGeOaR4t7O0?usp=sharing) |
| Ch5 | [link](https://colab.research.google.com/drive/1aK5flu1uj2z7KBLS1lzTraM2q_JXQsxb?usp=sharing) | [link](https://colab.research.google.com/drive/1lD6MFHj5AcqQpUNzVqA1QRjyNy5hjd2a?usp=sharing) |
| Ch6 | [link](https://colab.research.google.com/drive/12l-LgJG-Eq1UpICozQePVJTWxqjBvnhL?usp=drive_link) | [link](https://colab.research.google.com/drive/1MBLRuoVB8zCB7m46VLYNAeN27GWCqXBw?usp=sharing) |
| Ch7 | [link](https://colab.research.google.com/drive/1b6seb6KaI9-5WkAigfud6rB95W0fJOm1?usp=drive_link) | [link](https://colab.research.google.com/drive/1QuWatc898QvArL_eMb-6GcfZmNbzz_5V?usp=sharing) |
| Ch8 | [link](https://colab.research.google.com/drive/1vXKMZE62UlrOcQze8QzxJ0hA2JL5dakK?usp=drive_link) | [link](https://colab.research.google.com/drive/19aA4iWVP3nJWevLWUlyhgezVNB26UBB-?usp=sharing) |
| Ch9 | [link](https://colab.research.google.com/drive/1vGIGJZmjdFUbba8hyvdOGwYze7lmNMFI?usp=drive_link) | [link](https://colab.research.google.com/drive/123ituOmWqGFF3e2pS3TevfVPShM6Euzu?usp=sharing) |

## What we learned

## **Chapter 6: Dealing with Outliers**

I learned that outliers are data points that differ significantly from most values in a dataset and can affect data analysis and machine learning models. I understood how to detect outliers using the Z-score and Interquartile Range (IQR) methods. What surprised me was that a value like 100 could stand out from the other values but still not be detected as an outlier by the Z-score method because its Z-score remained within the range of -3 to 3. I also learned that outliers can be handled through capping, flooring, log transformation, or removal, depending on the situation.

## **Chapter 7: Feature Selection**

I learned that feature selection involves choosing the most relevant features to help a machine learning model make better predictions. I understood that correlation helps identify relationships between variables and that feature selection has three main methods: filter, wrapper, and embedded methods. What surprised me was that each method selects features differently. Filter methods use statistical measures, wrapper methods evaluate combinations based on model performance, and embedded methods select features during model training. This taught me that choosing relevant features can simplify a dataset and help improve model performance.

## **Chapter 8: Constructing a Preprocessing Pipeline**

I learned that a preprocessing pipeline organizes data preparation steps into a sequence that runs automatically. Using the Titanic dataset, I understood how missing values in the Age and Fare columns can be filled using the mean and how StandardScaler standardizes numerical values. I also learned that ColumnTransformer applies different preprocessing steps to selected columns. What surprised me was that these steps could be combined into one reusable process, making data preparation more consistent and reducing manual work before training a machine learning model.

## **Chapter 9: Real-World Application: Data Preprocessing**

I learned how to apply different preprocessing techniques to the Titanic dataset, including handling missing values, scaling numerical features, encoding categorical features, removing irrelevant columns, and grouping ages into categories through discretization. I understood that numerical features such as Age and Fare require different treatments from categorical features such as Sex and Embarked. What surprised me was how much preparation the dataset needed before it could be used for machine learning. I also learned that visualizations and data quality checks help me understand the effects of preprocessing and determine whether the data is ready for model training.

## Errors we found

## **Chapter 6: Dealing with Outliers**

**File**: Ch6.ipynb

**Error 1**: The Z-score method does not identify 100 as an outlier.

**Code cell 5**

The code uses np.abs(z_scores) > 3 to detect outliers. However, the Z-score for 100 is approximately 2.615, so the output is an empty array. This conflicts with the Markdown explanation that identifies 100 as a clear outlier.

**Correction**

Use the IQR method for this example, or adjust the Z-score threshold if justified. Do not assume that every outlier will have a Z-score above 3.
The IQR method in code cell 11 correctly identifies 100 as an outlier. The main issue in this chapter is the difference between the Z-score result and the written explanation.

## **Chapter 7: Feature Selection**

**File**: Ch7.ipynb

**Error 1**: The target variable is included in the selected features.

**Code cells 9 and 10**

The correlation calculation includes final grade, which is the target variable. Its correlation with itself is always 1.0, so the code includes it in relevant_features. This is misleading because the target is what the model should predict, not an input feature.

**Correction**

correlations = df_2.drop(columns='final grade').corrwith(
    df_2['final grade']
).sort_values()

relevant_features = correlations[correlations > 0.5]
print(relevant_features)

**Error 2**: The RFECV results are unreliable with this small dataset.

**Code cells 14 and 15**

The notebook uses five-fold cross-validation with only seven samples. Some validation folds contain too few samples to calculate the R² score reliably, which produces warnings. The selected feature may not generalize well.

**Correction**

Use a larger dataset. For this small demonstration, reduce the number of folds, while recognizing that the results will still be limited by the small sample size.
The LassoCV example also uses only seven samples. Its selected features should be treated as illustrative rather than reliable evidence of feature importance.

## **Chapter 8: Constructing a Preprocessing Pipeline**

**File**: Ch8.ipynb

**Issue 1**: The pipeline processes only Age and Fare.

**Code cells 15 to 17**

The ColumnTransformer applies imputation and scaling only to the Age and Fare columns. Other columns are dropped because remainder='drop' is the default setting. This is not a coding error if the goal is to demonstrate numerical preprocessing, but the transformed output does not contain the complete dataset.

**Correction**

If you want to retain other columns, define preprocessing for categorical features or use remainder='passthrough' when appropriate.
Issue 2: The notebook assumes the uploaded file is named train.csv.

**Code cell 5**

The line pd.read_csv('train.csv') will raise a FileNotFoundError if the uploaded file has a different name.

**Correction**

Confirm that the uploaded filename is train.csv, or use the filename returned by files.upload().
The main preprocessing steps are valid for the selected numerical columns. The important limitation is that the output contains only the transformed Age and Fare features.

## **Chapter 9: Real-World Application: Data Preprocessing**

File: Ch9.ipynb

Error 1: The original Age column is overwritten during discretization.

Code cell 21

data['Age'] = pd.cut(
    data['Age'],
    bins=[0, 12, 50, 200],
    labels=['Child', 'Adult', 'Elderly']
)

This replaces the original numerical ages with categories. As a result, the notebook cannot use data['Age'] later to plot the original age distribution.

Correction

data['AgeGroup'] = pd.cut(
    data['Age'],
    bins=[0, 12, 50, 200],
    labels=['Child', 'Adult', 'Elderly']
)

Keep the original Age column and store the categories in a separate column.
Error 2: The histogram uses the wrong column for the discretized ages.

Code cell 29

plt.hist(titanic_preprocessed[:, 2], alpha=0.5,
         label='After discretization')
         
The third column of titanic_preprocessed is not the discretized Age column. The transformed array starts with the scaled Age and Fare columns, followed by one-hot-encoded categorical features. Therefore, this plot shows the wrong data.

Correction

Plot the separate AgeGroup column directly, using a count plot or bar chart to show the number of passengers in each age group.
Error 3: The plot labeled Before discretization uses the modified Age column.

Code cell 28

Because code cell 21 overwrites Age, the code below does not plot the original numerical age distribution.
plt.hist(data['Age'].dropna(),
         alpha=0.5, label='Before discretization')
         
Correction

Save a copy of the original Age column before discretization, then use that copy for the original histogram.
Error 4: The age bins exclude age zero.

Code cell 21

By default, pd.cut() excludes the lowest boundary. With bins starting at zero, an age of exactly zero becomes a missing value.

Correction

data['AgeGroup'] = pd.cut(
    data['Age'],
    bins=[0, 12, 50, 200],
    labels=['Child', 'Adult', 'Elderly'],
    include_lowest=True
)

This includes zero in the first bin. If age zero must be handled correctly, confirm that the bin boundaries match the intended age groups.
One additional concern is code cell 33, which uses kde=True to plot Age after converting it to categorical labels. A kernel density estimate is intended for numerical data, so the plot should use the original numerical Age column or omit the KDE when plotting age categories.


## Note on AI tools

**Noble**
- The AI tools that used for this case study are Claude and ChatGPT; I've used these AI tools to verify the errors in each chapter that are stated in the Errors we found section in this repository. 

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
