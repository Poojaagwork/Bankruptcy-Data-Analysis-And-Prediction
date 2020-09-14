#!/usr/bin/env python
# coding: utf-8
# # FINAL PROJECT
# 
# ### Pooja Agrawal
# 
# 
# In[1]:
get_ipython().run_line_magic('reset', '-f')
# In[2]:
# To supress warnings
import warnings
warnings.filterwarnings("ignore")
# In[4]:
from pandas import ExcelWriter
from pandas import ExcelFile
from pandas import Series
from pandas.tools.plotting import scatter_matrix
from openpyxl import *
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy import stats
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
# In[5]:
#DATA LOADING 
yearone = pd.read_table("C:/Users/mailp/OneDrive/Documents/1year.csv", sep=",")
yeartwo = pd.read_table("C:/Users/mailp/OneDrive/Documents/2year.csv", sep=",")
yearthree = pd.read_table("C:/Users/mailp/OneDrive/Documents/3year.csv", sep=",")
yearfour = pd.read_table("C:/Users/mailp/OneDrive/Documents/4year.csv", sep=",")
yearfive = pd.read_table("C:/Users/mailp/OneDrive/Documents/5year.csv", sep=",")
# In[6]:
#Checking the datatype of coloumns 
print(yearone.dtypes)
print(yeartwo.dtypes)
print(yearthree.dtypes)
print(yearfour.dtypes)
print(yearfive.dtypes)
# In[7]:
#Converting datatype of columns into numeric 
yearone = yearone.convert_objects(convert_numeric=True)
yeartwo = yeartwo.convert_objects(convert_numeric=True)
yearthree = yearthree.convert_objects(convert_numeric=True)
yearfour = yearfour.convert_objects(convert_numeric=True)
yearfive = yearfive.convert_objects(convert_numeric=True)
# In[8]:
yearone
# In[9]:
#Counting the missing NAn for year one 
len(yearone.index)-yearone.count()
# In[10]:
#Cleaned Data for year 1
clean1 = yearone.dropna()
clean1
# In[11]:
#Descriptive analysis of year one 
clean1.describe()
# In[12]:
#Heatmap for year one (after dropping missing variables)
plt.figure(figsize=(20,20), dpi=200)
plt.title("HEATMAP FOR YEAR 1")
sns.heatmap(clean1.corr(), cmap='hot')
# In[13]:
# Separation of Data by bankrupty for year 1 
Bankrupt= clean1.loc[clean1["class"]== 1]
NonBankrupt = clean1.loc[clean1["class"]== 0]
# In[14]:
#Correlations for year 1 for Bankrupt Firms
plt.scatter(Bankrupt['Attr9'], Bankrupt['Attr34'], marker='+')
plt.xlabel("sales / total assets")
plt.ylabel("operating expenses / total liabilities")
plt.title("For Bankrupt Firm Year l")
plt.show()
# In[15]:
#Correlations for year 1 for NonBankrupt Firms
plt.scatter(NonBankrupt['Attr9'], NonBankrupt['Attr34'], marker='+')
plt.xlabel("sales / total assets")
plt.ylabel("operating expenses / total liabilities")
plt.title("For NonBankrupt Firm Year l")
plt.show()
# In[16]:
#Subsetting attributes dataset for year 1
raw_data_one = clean1.values
X_one = raw_data_one[:, 1:65]
#Subsetting class dataset for year 1
y_one = raw_data_one[:, 65]
# conversion of the class labels to integer-type array
y_one = y_one.astype(np.int64, copy=False)
print("Feature vectors of the dataset: ", "\n", X_one)
print("\n")
print("Labels of the dataset: ", "\n", y_one)
# In[17]:
# Feature Ranking using ExtraTreesClassifier For year 1
# Building an ExtraTrees Clasifier with 250 estimator for year 1
eT = ExtraTreesClassifier(n_estimators=250,random_state=42)
eT.fit(X_one, y_one)
# Compute the feature importances and sort for year 1 
importance = eT.feature_importances_
indices = np.argsort(importance)[::-1]
# Plot the feature importances of the forest for year 1 
# Change the range to select how many features to plot for year 1 
plt.figure()
plt.title("Feature importances for year 1")
plt.bar(range(0,64), importance[indices][:64], color="r", align="center")
plt.show()
# In[18]:
#Ranking and Printing the Attributes that affect bankrupty for year 1 
print("Index of features in order of decreasing importance for year 1: \n\n", indices)
finalAns1 = ""
for f in range(0,64):
    if importance[indices[f]] >= 0.025:
        finalAns1 += "," + str(indices[f]+1)
        
print("\nBest Attributes: " + finalAns1[1:])
# In[19]:
# so the best attributes are that are effecting bankrupty  for year 1 are 
#X27:profit on operating activities / financial expenses,
#X46:(current assets - inventory) / short-term liabilities,
#X34: operating expenses / total liabilities,
#X9: sales / total assets
#X15: (total liabilities * 365) / (gross profit + depreciation)
# In[20]:
#xx = clean1[["Attr27", "Attr46","Attr34","Attr9","Attr15"]]
# In[21]:
#subsetting best 5 attribues 
Bankrupt11 = Bankrupt[["Attr27", "Attr46","Attr34","Attr9","Attr15"]]
NonBankrupt11 = NonBankrupt[["Attr27", "Attr46","Attr34","Attr9","Attr15"]]
# In[22]:
Bankrupt11.describe()
# In[23]:
NonBankrupt11.describe()
# In[24]:
#t test for #X27:profit on operating activities / financial expenses for year 1
stats.ttest_ind(Bankrupt11[['Attr27',]], NonBankrupt11[['Attr27',]], equal_var = False)
# In[25]:
#t test for #X46:(current assets - inventory) / short-term liabilities for year 1
stats.ttest_ind(Bankrupt11[['Attr46',]], NonBankrupt11[['Attr46',]], equal_var = False)
# In[26]:
#t test for #X34: operating expenses / total liabilities for year 1 
stats.ttest_ind(Bankrupt11[['Attr34',]], NonBankrupt11[['Attr34',]], equal_var = False)
# In[27]:
#t test for #X9: sales / total assets for year 1
stats.ttest_ind(Bankrupt11[['Attr9',]], NonBankrupt11[['Attr9',]], equal_var = False)
# In[28]:
#t test for #X15: (total liabilities * 365) / (gross profit + depreciation) for year 1
stats.ttest_ind(Bankrupt11[['Attr15',]], NonBankrupt11[['Attr15',]], equal_var = False)
# In[29]:
#scatter plot of 5 attributes for year 1 for bankrupt firm 
plt.clf()
scatter_plot11 = scatter_matrix(Bankrupt11, alpha=1, figsize=(11, 11))
# In[30]:
#scatter plot of 5 attributes for year 1 for Non-bankrupt firm 
scatter_matrix(NonBankrupt11, alpha=0.8, figsize=(11,11))
plt.show()
# In[31]:
#Regression
#Calculate the proportion of two classes in Year 1 dataset
# Counting the number of two classes in Year 1
target_count = clean1['class'].value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target) for year 1');
# In[32]:
#Split the dataset into training and test dataset. We use 80-20 split. Stratified
test_percent = 0.20
random_seed = 42
X_year1_train, X_year1_test, y_year1_train, y_year1_test = train_test_split(X_one, 
                                                                            y_one, 
                                                                            test_size=test_percent, 
                                                                            random_state=random_seed,
                                                                            stratify = y_one)
print("No. of samples in Traning Dataset: ", X_year1_train.shape[0])
print("No. of samples in Test Dataset: ", X_year1_test.shape[0])
# In[33]:
#Count class breakdown after split
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year1_train).count(l) / y_year1_train.shape[0]))
print('\nTest Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year1_test).count(l) / y_year1_test.shape[0]))
# In[34]:
#SMOTE oversampling in training dataset
os = SMOTE(random_state=42)
X_year1_train_os, y_year1_train_os=os.fit_sample(X_year1_train, y_year1_train)
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year1_train_os).count(l) / y_year1_train_os.shape[0]))
# In[35]:
#OLS Regression 
import statsmodels.api as sm
ols_model=sm.OLS(y_year1_train_os,X_year1_train_os[:,33])
results=ols_model.fit()
print(results.summary())
# In[36]:
#residuals
df = pd.DataFrame(results.resid)
df.describe()
# In[37]:
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot
#plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)
# In[38]:
# fitted values (need a constant term for intercept)
model_fitted_y = results.fittedvalues
# model residuals
model_residuals = results.resid
# normalized residuals
model_norm_residuals = results.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = results.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = results.get_influence().cooks_distance[0]
# In[39]:
aX = np.asarray(X_year1_train_os[:,33])
aY = np.asarray(y_year1_train_os)
plot_lm_1 = plt.figure(1)
#plot_lm_1.set_figheight(8)
#plot_lm_1.set_figwidth(12)
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, aY)
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plt.show()
# In[40]:
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
#plot_lm_2.set_figheight(8)
#plot_lm_2.set_figwidth(12)
plot_lm_2.axes[0].set_title('Normal Q-Q (operating expenses / total liabilities VS class)')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
plt.show()
# In[41]:
plot_lm_3 = plt.figure(3)
#plot_lm_3.set_figheight(8)
#plot_lm_3.set_figwidth(12)
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
plot_lm_3.axes[0].set_title('Scale-Location(operating expenses / total liabilities VS class)')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
plt.show()
# In[42]:
plot_lm_4 = plt.figure(4)
#plot_lm_4.set_figheight(8)
#plot_lm_4.set_figwidth(12)
influence = results.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
plt.stem(np.arange(len(c)), c, markerfmt=",")
plot_lm_4.axes[0].set_xlim(0, 9)
plot_lm_4.axes[0].set_ylim(0,.0007)
plot_lm_4.axes[0].set_title('Cook Distance Plot (operating expenses / total liabilities VS class)')
plot_lm_4.axes[0].set_xlabel('Cook Distance')
plot_lm_4.axes[0].set_ylabel('Observation')
plt.show()
# In[185]:
# Make predictions on Test dataset using LogisticRegression for year 1
lr = LogisticRegression()
lr.fit(X_year1_train_os, y_year1_train_os)
predictionsLR = lr.predict(X_year1_test)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score for year 1:')
print(accuracy_score(y_year1_test, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(y_year1_test, predictionsLR))
print('Classification Report for year 1:')
print(classification_report(y_year1_test, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')
# In[44]:
#ROC Curve for year 1
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_year1_test, lr.predict(X_year1_test))
fpr, tpr, thresholds = roc_curve(y_year1_test, lr.predict_proba(X_year1_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for year 1')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# # ----------------------------------------------  YEAR 2  ------------------------------------------------
# In[45]:
yeartwo 
# In[46]:
#cleaned Data for year 2
clean2 = yeartwo.dropna()
clean2
# In[47]:
#Descriptive analysis of year two
clean2.describe()
# In[48]:
#Heatmap for year two (after dropping missing variables)
plt.figure(figsize=(20,20), dpi=200)
plt.title("HEATMAP FOR YEAR 2")
sns.heatmap(clean2.corr(), cmap='hot')
# In[49]:
# Separation of Data by bankrupty for year 2
Bankruptx= clean2.loc[clean2["class"]== 1]
NonBankrupty = clean2.loc[clean2["class"]== 0]
# In[50]:
#Correlations for year 2 for Bankrupt Firms
plt.scatter(Bankruptx['Attr46'], Bankruptx['Attr34'], marker='+')
plt.xlabel("current assets - inventory) / short-term liabilities")
plt.ylabel("operating expenses / total liabilities")
plt.title("For Bankrupt Firm Year 2")
plt.show()
# In[51]:
#Correlations for year 2 for NonBankrupt Firms
plt.scatter(NonBankrupty['Attr46'], NonBankrupty['Attr34'], marker='+')
plt.xlabel("(current assets - inventory) / short-term liabilities")
plt.ylabel("operating expenses / total liabilities")
plt.title("For NonBankrupt Firm Year 2")
plt.show()
# In[52]:
#Subsetting attributes dataset for year 2
raw_data_two = clean2.values
x_two = raw_data_two[:, 1:65]
#Subsetting class dataset for year 2 
y_two = raw_data_two[:, 65]
# conversion of the class labels to integer-type array
y_two = y_two.astype(np.int64, copy=False)
print("Feature vectors of the dataset: ", "\n", x_two)
print("\n")
print("Labels of the dataset: ", "\n", y_two)
# In[53]:
# Feature Ranking using ExtraTreesClassifier For year 2
# Building an ExtraTrees Clasifier with 250 estimator for year 2
eT2 = ExtraTreesClassifier(n_estimators=250,random_state=42)
eT2.fit(x_two, y_two)
# Compute the feature importances and sort for year 2 
importance2 = eT2.feature_importances_
indices2 = np.argsort(importance2)[::-1]
# Plot the feature importances of the forest for year 2
# Change the range to select how many features to plot for year 2
plt.figure()
plt.title("Feature importances for year 2")
plt.bar(range(0,64), importance2[indices2][:64], color="r", align="center")
plt.show()
# In[54]:
#Ranking and Printing the Attributes that affect bankrupty for year 2 
print("Index of features in order of decreasing importance for year 2: \n\n", indices2)
finalAns2 = ""
for f in range(0,64):
    if importance2[indices2[f]] >= 0.022:
        finalAns2 += "," + str(indices2[f]+1)
        
print("\nBest Attributes: " + finalAns2[1:])
# In[55]:
# so the best 5 attributes are that are effecting bankrupty  for year 2 are 
#X46:(current assets - inventory) / short-term liabilities,
#X34: operating expenses / total liabilities,
#X27 profit on operating activities / financial expenses
#X24 gross profit (in 3 years) / total assets
#X58 total costs /total sales
# In[56]:
#subsetting best 5 attribues 
Bankrupt22 = Bankruptx[["Attr46","Attr34", "Attr27", "Attr24", "Attr58"]]
NonBankrupt22 = NonBankrupty[["Attr46","Attr34", "Attr27", "Attr24", "Attr58"]]
# In[57]:
Bankrupt22.describe()
# In[58]:
NonBankrupt22.describe()
# In[59]:
#t test for #X46:(current assets - inventory) / short-term liabilities for year 2 
stats.ttest_ind(Bankrupt22[['Attr46',]], NonBankrupt22[['Attr46',]], equal_var = False)
# In[60]:
#t test for #X34: operating expenses / total liabilities for year 2 
stats.ttest_ind(Bankrupt22[['Attr34',]], NonBankrupt22[['Attr34',]], equal_var = False)
# In[61]:
#t test for #X27:profit on operating activities / financial expenses for year 2
stats.ttest_ind(Bankrupt22[['Attr27',]], NonBankrupt22[['Attr27',]], equal_var = False)
# In[62]:
#t test for #X24 gross profit (in 3 years) / total assets for year 2
stats.ttest_ind(Bankrupt22[['Attr24',]], NonBankrupt22[['Attr24',]], equal_var = True)
# In[63]:
#t test for #X58 total costs /total sales for year 2
stats.ttest_ind(Bankrupt22[['Attr58',]], NonBankrupt22[['Attr58',]], equal_var = False)
# In[177]:
scatter_matrix(Bankrupt22, alpha=0.8, figsize=(11,11))
plt.show()
# In[178]:
scatter_matrix(NonBankrupt22, alpha=0.8, figsize=(11,11))
plt.show()
# In[68]:
Bankrupt2 = Bankruptx[[ "Attr46","Attr34","Attr24"]]
NonBankrupt2 = NonBankrupty[[ "Attr46","Attr34","Attr24"]]
# In[71]:
# Counting the number of two classes in Year 2
target_count2 = clean2['class'].value_counts()
print('Class 0:', target_count2[0])
print('Class 1:', target_count2[1])
print('Proportion:', round(target_count2[0] / target_count2[1], 2), ': 1')
target_count2.plot(kind='bar', title='Count (target) for year 2');
# In[72]:
#Split the dataset into training and test sets for year 2 
test_percent = 0.20
random_seed = 42
X_year2_train, X_year2_test, y_year2_train, y_year2_test = train_test_split(x_two, 
                                                                            y_two, 
                                                                            test_size=test_percent, 
                                                                            random_state=random_seed,
                                                                            stratify = y_two)
print("No. of samples in Traning Dataset: ", X_year2_train.shape[0])
print("No. of samples in Test Dataset: ", X_year2_test.shape[0])
# In[73]:
#Count class breakdown after spilt for year 2
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year2_train).count(l) / y_year2_train.shape[0]))
print('\nTest Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year2_test).count(l) / y_year2_test.shape[0]))
# In[74]:
#Smote sampling for year 2 
os = SMOTE(random_state=42)
X_year2_train_os, y_year2_train_os=os.fit_sample(X_year2_train, y_year2_train)
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year2_train_os).count(l) / y_year2_train_os.shape[0]))
# In[75]:
#OlS Regression for year 2
import statsmodels.api as sm
ols_model2=sm.OLS(y_year2_train_os,X_year2_train_os[:,33])
ols_result2=ols_model2.fit()
print(ols_result2.summary())
# In[186]:
# Make predictions on Test dataset using LogisticRegression for year 2
lr = LogisticRegression()
lr.fit(X_year2_train_os, y_year2_train_os)
predictionsLR = lr.predict(X_year2_test)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score for year 2:')
print(accuracy_score(y_year2_test, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(y_year2_test, predictionsLR))
print('Classification Report for year 2:')
print(classification_report(y_year2_test, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')
# In[77]:
#ROC Curve for year 2
logit_roc_auc = roc_auc_score(y_year2_test, lr.predict(X_year2_test))
fpr, tpr, thresholds = roc_curve(y_year2_test, lr.predict_proba(X_year2_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for year 2')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# # ----------------------------------------------  YEAR 3  ------------------------------------------------
# In[78]:
yearthree
# In[79]:
#cleaned Data for year 3
clean3 = yearthree.dropna()
clean3
# In[80]:
#Descriptive analysis of year three
clean3.describe()
# In[81]:
#Heatmap for year three (after dropping missing variables)
plt.figure(figsize=(20,20), dpi=200)
plt.title("HEATMAP FOR YEAR 3")
sns.heatmap(clean3.corr(), cmap='hot')
# In[82]:
# Separation of Data by bankrupty for year 3
Bankruptxx= clean3.loc[clean3["class"]== 1]
NonBankruptyy = clean3.loc[clean3["class"]== 0]
# In[83]:
#Correlations for year 3 for Bankrupt Firms
plt.scatter(Bankruptxx['Attr58'], Bankruptxx['Attr34'], marker='+')
plt.xlabel("total costs /total sales")
plt.ylabel("operating expenses / total liabilities")
plt.title("For Bankrupt Firm Year 3")
plt.show()
# In[84]:
#Correlations for year 3 for NonBankrupt Firms
plt.scatter(NonBankruptyy['Attr58'], NonBankruptyy['Attr34'], marker='+')
plt.xlabel("total costs /total sales")
plt.ylabel("operating expenses / total liabilities")
plt.title("For NonBankrupt Firm Year 3")
plt.show()
# In[85]:
#Subsetting attributes dataset for year 3
raw_data_three = clean3.values
x_three = raw_data_three[:, 1:65]
#Subsetting class dataset for year 3
y_three = raw_data_three[:, 65]
# conversion of the class labels to integer-type array
y_three = y_three.astype(np.int64, copy=False)
print("Feature vectors of the dataset: ", "\n", x_three)
print("\n")
print("Labels of the dataset: ", "\n", y_three)
# In[86]:
# Feature Ranking using ExtraTreesClassifier For year 3
# Building an ExtraTrees Clasifier with 250 estimator for year 3
eT3 = ExtraTreesClassifier(n_estimators=250,random_state=42)
eT3.fit(x_three, y_three)
# Compute the feature importances and sort for year 3
importance3 = eT3.feature_importances_
indices3 = np.argsort(importance3)[::-1]
# Plot the feature importances of the forest for year 3
# Change the range to select how many features to plot for year 3
plt.figure()
plt.title("Feature importances for year 3")
plt.bar(range(0,64), importance3[indices3][:64], color="r", align="center")
plt.show()
# In[87]:
#Ranking and Printing the Attributes that affect bankrupty for year 3
print("Index of features in order of decreasing importance for year 3: \n\n", indices3)
finalAns3 = ""
for f in range(0,64):
    if importance3[indices3[f]] >= 0.023:
        finalAns3 += "," + str(indices3[f]+1)
        
print("\nBest Attributes: " + finalAns3[1:])
# In[88]:
# so the best 5 attributes are that are effecting bankrupty  for year 3 are
#X15 (total liabilities * 365) / (gross profit + depreciation)
#X46:(current assets - inventory) / short-term liabilities,
#X24 gross profit (in 3 years) / total assets,
#X34: operating expenses / total liabilities,
#X58 total costs /total sales
# In[89]:
#Subsetting 5 best atttibutes 
Bankrupt33 = Bankruptxx[["Attr15","Attr46","Attr34", "Attr24", "Attr58"]]
NonBankrupt33 = NonBankruptyy[["Attr15","Attr46","Attr34", "Attr24", "Attr58"]]
# In[90]:
Bankrupt33.describe()
# In[91]:
NonBankrupt33.describe()
# In[92]:
#t test for #X15: (total liabilities * 365) / (gross profit + depreciation) for year 3
stats.ttest_ind(Bankrupt33[['Attr15',]], NonBankrupt33[['Attr15',]], equal_var = False)
# In[93]:
#t test for #X46:(current assets - inventory) / short-term liabilities for year 3
stats.ttest_ind(Bankrupt33[['Attr46',]], NonBankrupt33[['Attr46',]], equal_var = False)
# In[94]:
#t test for #X34: operating expenses / total liabilities for year 3
stats.ttest_ind(Bankrupt33[['Attr34',]], NonBankrupt33[['Attr34',]], equal_var = False)
# In[95]:
#t test for #X24 gross profit (in 3 years) / total assets for year 3
stats.ttest_ind(Bankrupt33[['Attr24',]], NonBankrupt33[['Attr24',]], equal_var = False)
# In[96]:
#t test for #X58 total costs /total sales for year 3
stats.ttest_ind(Bankrupt33[['Attr58',]], NonBankrupt33[['Attr58',]], equal_var = False)
# In[179]:
scatter_matrix(Bankrupt33, alpha=0.8, figsize=(11,11))
plt.show()
# In[180]:
scatter_matrix(NonBankrupt22, alpha=0.8, figsize=(11,11))
plt.show()
# In[101]:
Bankrupt3 = Bankruptxx[[ "Attr46","Attr34","Attr24"]]
NonBankrupt3 = NonBankruptyy[[ "Attr46","Attr34","Attr24"]]
# In[104]:
# Counting the number of two classes in Year 3
target_count3 = clean3['class'].value_counts()
print('Class 0:', target_count3[0])
print('Class 1:', target_count3[1])
print('Proportion:', round(target_count3[0] / target_count3[1], 2), ': 1')
target_count3.plot(kind='bar', title='Count (target) for year 3');
# In[105]:
#Split the dataset into training and test sets for year 3
test_percent = 0.20
random_seed = 42
X_year3_train, X_year3_test, y_year3_train, y_year3_test = train_test_split(x_three, 
                                                                            y_three, 
                                                                            test_size=test_percent, 
                                                                            random_state=random_seed,
                                                                            stratify = y_three)
print("No. of samples in Traning Dataset: ", X_year3_train.shape[0])
print("No. of samples in Test Dataset: ", X_year3_test.shape[0])
# In[106]:
#Count class breakdown after spilt for year 3
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year3_train).count(l) / y_year3_train.shape[0]))
print('\nTest Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year3_test).count(l) / y_year3_test.shape[0]))
# In[107]:
#Smote sampling for year 3
os = SMOTE(random_state=42)
X_year3_train_os, y_year3_train_os=os.fit_sample(X_year3_train, y_year3_train)
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year3_train_os).count(l) / y_year3_train_os.shape[0]))
# In[108]:
#OlS Regression
import statsmodels.api as sm
ols_model3=sm.OLS(y_year3_train_os,X_year3_train_os)
ols_result3=ols_model3.fit()
print(ols_result3.summary())
# In[187]:
# Make predictions on Test dataset using LogisticRegression for year 3
lr = LogisticRegression()
lr.fit(X_year3_train_os, y_year3_train_os)
predictionsLR = lr.predict(X_year3_test)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score for year 3:')
print(accuracy_score(y_year3_test, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(y_year3_test, predictionsLR))
print('Classification Report for year 3:')
print(classification_report(y_year3_test, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')
# In[110]:
#ROC Curve for year 3
logit_roc_auc = roc_auc_score(y_year3_test, lr.predict(X_year3_test))
fpr, tpr, thresholds = roc_curve(y_year3_test, lr.predict_proba(X_year3_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for year 3')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# # ----------------------------------------------  YEAR 4  ------------------------------------------------
# In[111]:
yearfour
# In[112]:
#cleaned Data for year 4
clean4 = yearfour.dropna()
clean4
# In[113]:
#Descriptive analysis of year four
clean4.describe()
# In[114]:
#Heatmap for year four (after dropping missing variables)
plt.figure(figsize=(20,20), dpi=200)
plt.title("HEATMAP FOR YEAR 4")
sns.heatmap(clean4.corr(), cmap='hot')
# In[115]:
# Separation of Data by bankrupty for year 4
Bankruptx4= clean4.loc[clean4["class"]== 1]
NonBankrupty4 = clean4.loc[clean4["class"]== 0]
# In[116]:
#Correlations for year 4 for Bankrupt Firms
plt.scatter(Bankruptx4['Attr46'], Bankruptx4['Attr24'], marker='+')
plt.xlabel("current assets - inventory) / short-term liabilities")
plt.ylabel("gross profit (in 3 years) / total assets")
plt.title("For Bankrupt Firm Year 4")
plt.show()
# In[117]:
#Correlations for year 4 for NonBankrupt Firms
plt.scatter(NonBankrupty4['Attr46'], NonBankrupty4['Attr24'], marker='+')
plt.xlabel("current assets - inventory) / short-term liabilities")
plt.ylabel("gross profit (in 3 years) / total assets")
plt.title("For NonBankrupt Firm Year 4")
plt.show()
# In[118]:
#Subsetting attributes dataset for year 4
raw_data_four = clean4.values
x_four = raw_data_four[:, 1:65]
#Subsetting class dataset for year 4
y_four = raw_data_four[:, 65]
# conversion of the class labels to integer-type array
y_four = y_four.astype(np.int64, copy=False)
print("Feature vectors of the dataset: ", "\n", x_four)
print("\n")
print("Labels of the dataset: ", "\n", y_four)
# In[119]:
# Feature Ranking using ExtraTreesClassifier For year 4
# Building an ExtraTrees Clasifier with 250 estimator for year 4
eT4 = ExtraTreesClassifier(n_estimators=250,random_state=42)
eT4.fit(x_four, y_four)
# Compute the feature importances and sort for year 4
importance4 = eT4.feature_importances_
indices4 = np.argsort(importance4)[::-1]
# Plot the feature importances of the forest for year 4
# Change the range to select how many features to plot for year 4
plt.figure()
plt.title("Feature importances for year 4")
plt.bar(range(0,64), importance4[indices4][:64], color="r", align="center")
plt.show()
# In[120]:
#Ranking and Printing the Attributes that affect bankrupty for year 4
print("Index of features in order of decreasing importance for year 4: \n\n", indices4)
finalAns4 = ""
for f in range(0,64):
    if importance4[indices4[f]] >= 0.020:
        finalAns4 += "," + str(indices4[f]+1)
        
print("\nBest Attributes: " + finalAns4[1:])
# In[121]:
# so the best 5 attributes are that are effecting bankrupty  for year 4 are 
#X24 gross profit (in 3 years) / total assets,
#X34: operating expenses / total liabilities,
#X58 total costs /total sales,
#X46:(current assets - inventory) / short-term liabilities,
#X15 (total liabilities * 365) / (gross profit + depreciation)
# In[122]:
#Seperation of best 5 attributes for year 4
Bankrupt44 = Bankruptx4[["Attr46","Attr34", "Attr15", "Attr24", "Attr58"]]
NonBankrupt44 = NonBankrupty4[["Attr46","Attr34", "Attr15", "Attr24", "Attr58"]]
# In[123]:
Bankrupt44.describe()
# In[124]:
NonBankrupt44.describe()
# In[125]:
#t test for #X46:(current assets - inventory) / short-term liabilities for year 4 
stats.ttest_ind(Bankrupt44[['Attr46',]], NonBankrupt44[['Attr46',]], equal_var = False)
# In[126]:
#t test for #X34: operating expenses / total liabilities for year 4
stats.ttest_ind(Bankrupt44[['Attr34',]], NonBankrupt44[['Attr34',]], equal_var = False)
# In[127]:
#t test for #X15: (total liabilities * 365) / (gross profit + depreciation) for year 4
stats.ttest_ind(Bankrupt44[['Attr15',]], NonBankrupt44[['Attr15',]], equal_var = False)
# In[128]:
#t test for #X24 gross profit (in 3 years) / total assets for year 4
stats.ttest_ind(Bankrupt44[['Attr24',]], NonBankrupt44[['Attr24',]], equal_var = True)
# In[129]:
#t test for #X58 total costs /total sales for year 4
stats.ttest_ind(Bankrupt44[['Attr58',]], NonBankrupt44[['Attr58',]], equal_var = False)
# In[181]:
scatter_matrix(Bankrupt44, alpha=0.8, figsize=(11,11))
plt.show()
# In[182]:
scatter_matrix(NonBankrupt44, alpha=0.8, figsize=(11,11))
plt.show()
# In[134]:
Bankrupt4 = Bankruptx4[[ "Attr46","Attr34"]]
NonBankrupt4 = NonBankrupty4[[ "Attr46","Attr34"]]
# In[137]:
# Counting the number of two classes in Year 4
target_count4 = clean4['class'].value_counts()
print('Class 0:', target_count4[0])
print('Class 1:', target_count4[1])
print('Proportion:', round(target_count4[0] / target_count4[1], 2), ': 1')
target_count4.plot(kind='bar', title='Count (target) for year 4');
# In[138]:
#Split the dataset into training and test sets for year 4
test_percent = 0.20
random_seed = 42
X_year4_train, X_year4_test, y_year4_train, y_year4_test = train_test_split(x_four, 
                                                                            y_four, 
                                                                            test_size=test_percent, 
                                                                            random_state=random_seed,
                                                                            stratify = y_four)
print("No. of samples in Traning Dataset: ", X_year4_train.shape[0])
print("No. of samples in Test Dataset: ", X_year4_test.shape[0])
# In[139]:
#Count class breakdown after spilt for year 4
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year4_train).count(l) / y_year4_train.shape[0]))
print('\nTest Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year4_test).count(l) / y_year4_test.shape[0]))
# In[140]:
#Smote sampling for year 4 
os = SMOTE(random_state=42)
X_year4_train_os, y_year4_train_os=os.fit_sample(X_year4_train, y_year4_train)
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year4_train_os).count(l) / y_year4_train_os.shape[0]))
# In[141]:
#OlS Regression for year 4
import statsmodels.api as sm
ols_model4=sm.OLS(y_year4_train_os,X_year4_train_os)
ols_result4=ols_model4.fit()
print(ols_result4.summary())
# In[188]:
# Make predictions on Test dataset using LogisticRegression for year 4
lr = LogisticRegression()
lr.fit(X_year4_train_os, y_year4_train_os)
predictionsLR = lr.predict(X_year4_test)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score for year 4:')
print(accuracy_score(y_year4_test, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(y_year4_test, predictionsLR))
print('Classification Report for year 4:')
print(classification_report(y_year4_test, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')
# In[143]:
#ROC Curve for year 4
logit_roc_auc = roc_auc_score(y_year4_test, lr.predict(X_year4_test))
fpr, tpr, thresholds = roc_curve(y_year4_test, lr.predict_proba(X_year4_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for year 4')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# # ----------------------------------------------  YEAR 5  ------------------------------------------------
# In[144]:
yearfive
# In[145]:
#cleaned Data for year 5
clean5 = yearfive.dropna()
clean5
# In[146]:
#Descriptive analysis of year five
clean5.describe()
# In[147]:
#Heatmap for year five (after dropping missing variables)
plt.figure(figsize=(20,20), dpi=200)
plt.title("HEATMAP FOR YEAR 5")
sns.heatmap(clean5.corr(), cmap='hot')
# In[148]:
# Separation of Data by bankrupty for year 5
Bankruptx5= clean5.loc[clean5["class"]== 1]
NonBankrupty5 = clean5.loc[clean5["class"]== 0]
# In[149]:
#Correlations for year 5 for Bankrupt Firms
plt.scatter(Bankruptx5['Attr39'], Bankruptx5['Attr56'], marker='+')
plt.xlabel("profit on sales / sales")
plt.ylabel("(sales - cost of products sold) / sales")
plt.title("For Bankrupt Firm Year 5")
plt.show()
# In[150]:
#Correlations for year 5 for Bankrupt Firms
plt.scatter(NonBankrupty5['Attr39'], NonBankrupty5['Attr56'], marker='+')
plt.xlabel("profit on sales / sales")
plt.ylabel("(sales - cost of products sold) / sales")
plt.title("For NonBankrupt Firm Year 5")
plt.show()
# In[151]:
#Subsetting attributes dataset for year 5
raw_data_five = clean5.values
x_five = raw_data_five[:, 1:65]
#Subsetting class dataset for year 5
y_five = raw_data_five[:, 65]
# conversion of the class labels to integer-type array
y_five = y_five.astype(np.int64, copy=False)
print("Feature vectors of the dataset: ", "\n", x_five)
print("\n")
print("Labels of the dataset: ", "\n", y_five)
# In[152]:
# Feature Ranking using ExtraTreesClassifier For year 5
# Building an ExtraTrees Clasifier with 250 estimator for year 5
eT5 = ExtraTreesClassifier(n_estimators=250,random_state=42)
eT5.fit(x_five, y_five)
# Compute the feature importances and sort for year 5
importance5 = eT5.feature_importances_
indices5 = np.argsort(importance5)[::-1]
# Plot the feature importances of the forest for year 5
# Change the range to select how many features to plot for year 5
plt.figure()
plt.title("Feature importances for year 5")
plt.bar(range(0,64), importance5[indices5][:64], color="r", align="center")
plt.show()
# In[153]:
#Ranking and Printing the Attributes that affect bankrupty for year 5
print("Index of features in order of decreasing importance for year 5: \n\n", indices4)
finalAns5 = ""
for f in range(0,64):
    if importance5[indices5[f]] >= 0.021:
        finalAns5 += "," + str(indices5[f]+1)
        
print("\nBest Attributes: " + finalAns5[1:])
# In[154]:
# so the best 5 attributes are that are effecting bankrupty  for year 5 are 
#X24 gross profit (in 3 years) / total assets,
#X34: operating expenses / total liabilities,
#X35 profit on sales / total assets
#X39 profit on sales / sales
#X56 (sales - cost of products sold) / sales
# In[155]:
#Subsetting of best 5 attributes for year 5
Bankrupt55 = Bankruptx5[["Attr24","Attr34", "Attr35", "Attr39", "Attr56"]]
NonBankrupt55 = NonBankrupty5[["Attr24","Attr34", "Attr35", "Attr39", "Attr56"]]
# In[156]:
Bankrupt55.describe()
# In[157]:
NonBankrupt55.describe()
# In[158]:
#t test for #X24 gross profit (in 3 years) / total assets for year 5
stats.ttest_ind(Bankrupt55[['Attr24',]], NonBankrupt55[['Attr24',]], equal_var = True)
# In[159]:
#t test for #X34: operating expenses / total liabilities for year 5
stats.ttest_ind(Bankrupt55[['Attr34',]], NonBankrupt55[['Attr34',]], equal_var = False)
# In[160]:
#t test for #X35: profit on sales / total assets for year 5
stats.ttest_ind(Bankrupt55[['Attr35',]], NonBankrupt55[['Attr35',]], equal_var = False)
# In[161]:
#t test for #X39: profit on sales / sales
stats.ttest_ind(Bankrupt55[['Attr24',]], NonBankrupt55[['Attr24',]], equal_var = True)
# In[162]:
#t test for #X56 (sales - cost of products sold) / sales for year 5
stats.ttest_ind(Bankrupt55[['Attr56',]], NonBankrupt55[['Attr56',]], equal_var = False)
# In[183]:
scatter_matrix(Bankrupt55, alpha=0.8, figsize=(11,11))
plt.show()
# In[184]:
scatter_matrix(NonBankrupt55, alpha=0.8, figsize=(11,11))
plt.show()
# In[167]:
Bankrupt5 = Bankruptx5[[ "Attr56","Attr34","Attr35"]]
NonBankrupt5 = NonBankrupty5[[ "Attr56","Attr34","Attr35"]]
# In[170]:
# Counting the number of two classes in Year 5
target_count5 = clean5['class'].value_counts()
print('Class 0:', target_count5[0])
print('Class 1:', target_count5[1])
print('Proportion:', round(target_count5[0] / target_count5[1], 2), ': 1')
target_count5.plot(kind='bar', title='Count (target) for year 5');
# In[171]:
#Split the dataset into training and test sets for year 5
test_percent = 0.20
random_seed = 42
X_year5_train, X_year5_test, y_year5_train, y_year5_test = train_test_split(x_five, 
                                                                            y_five, 
                                                                            test_size=test_percent, 
                                                                            random_state=random_seed,
                                                                            stratify = y_five)
print("No. of samples in Traning Dataset: ", X_year5_train.shape[0])
print("No. of samples in Test Dataset: ", X_year5_test.shape[0])
# In[172]:
#Count class breakdown after spilt for year 5
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year5_train).count(l) / y_year5_train.shape[0]))
print('\nTest Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year5_test).count(l) / y_year5_test.shape[0]))
# In[173]:
#Smote sampling for year 5 
os = SMOTE(random_state=42)
X_year5_train_os, y_year5_train_os=os.fit_sample(X_year5_train, y_year5_train)
print('Class label frequencies')
print('\nTraining Dataset:')
for l in range(0, 2):
    print('Class {:} samples: {:.2%}'.format(l, list(y_year5_train_os).count(l) / y_year5_train_os.shape[0]))
# In[174]:
#OlS Regression for year 5
import statsmodels.api as sm
ols_model5=sm.OLS(y_year5_train_os,X_year5_train_os)
ols_result5=ols_model5.fit()
print(ols_result5.summary())
# In[189]:
# Make predictions on Test dataset using LogisticRegression for year 5
lr = LogisticRegression()
lr.fit(X_year5_train_os, y_year5_train_os)
predictionsLR = lr.predict(X_year5_test)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score for year 5:')
print(accuracy_score(y_year5_test, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(y_year5_test, predictionsLR))
print('Classification Report for year 5:')
print(classification_report(y_year5_test, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')
# In[176]:
#ROC Curve for year 5
logit_roc_auc = roc_auc_score(y_year5_test, lr.predict(X_year5_test))
fpr, tpr, thresholds = roc_curve(y_year5_test, lr.predict_proba(X_year5_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for year 5')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()