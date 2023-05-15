# library doc string
"""
library of functions to find customers who are likely to churn.
author: Younes Kebour
Date: May. 15th 2023

"""

# import libraries
#import shap
import os
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe

    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.title("Churn distribution")
    plt.savefig('images/eda/churn_hist.png', format='png')
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.title("Churns Age distribution")
    plt.savefig('images/eda/Customer_Age_hist.png', format='png')
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Marital_Status_bar_plot.png', format='png')
    plt.close()

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct.png', format='png')
    plt.close()

    quant_columns = dataframe[
        ['Customer_Age',
         'Dependent_count',
         'Months_on_book',
         'Total_Relationship_Count',
         'Months_Inactive_12_mon',
         'Contacts_Count_12_mon',
         'Credit_Limit',
         'Total_Revolving_Bal',
         'Avg_Open_To_Buy',
         'Total_Amt_Chng_Q4_Q1',
         'Total_Trans_Amt',
         'Total_Trans_Ct',
         'Total_Ct_Chng_Q4_Q1',
         'Avg_Utilization_Ratio']
    ]
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        quant_columns.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig('images/eda/heatmap.png', format='png')
    plt.close()


def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        # gender encoded column

        col_groups = dataframe.groupby(col).mean(numeric_only=True)[response]
        new_col = col + '_' + response
        col_lst = []
        for val in dataframe[col]:
            col_lst.append(col_groups.loc[val])
        dataframe[new_col] = col_lst
    #df.drop(category_lst, axis=1, inplace=True)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    data engineering fucntion to convert categorical feature aloand generate train and test dataset

    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    new_dataframe = encoder_helper(dataframe, cat_columns)
    y = new_dataframe['Churn']
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = new_dataframe[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder using plot_classification_report
    helper function

    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''

    plt.rc('figure', figsize=(10, 10))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        'images/results/Classification_report_Random_Forest.png',
        format='png')
    plt.close()

    plt.rc('figure', figsize=(10, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        'images/results/Classification_report_Logistic_Regression.png',
        format='png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig('images/results/feature_importance_plot.png', format='png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    '''
    # plot roc
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        y_test,
        cv_rfc.best_estimator_.predict( X_test,),


        ax=ax,
    )

    plot_roc_curve(y_test,lrc.predict(X_test), ax=ax)
    '''
    # define metrics
    y_pred_proba = cv_rfc.best_estimator_.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('images/results/LRC_ROC_Curves.png', format='png')
    plt.close()
    # define metrics
    y_pred_proba = lrc.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig('images/results/RFC_ROC_Curves.png', format='png')
    plt.close()

    plt.close(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    '''
    feature_importance_plot(cv_rfc, X_train, "./images/results")

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    plt.savefig('images/results/shap_plot.png', format='png')
    plt.close()
    '''


if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    y_test = y_test.astype(int)
    print('training starting ....')
    train_models(X_train, X_test, y_train, y_test)
    print('Training completed')
