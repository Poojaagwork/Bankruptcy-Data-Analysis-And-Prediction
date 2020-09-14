The dataset I used for this analysis comes from Poland. During the 1990s, the Polish economy transitioned into a capitalist economy. Since not much time has passed since then, any bankruptcy model developed for a developed economy cannot be applied directly to Poland without modifications. This dataset is the result of an effort to create a large dataset to study bankruptcy in Polish companies.

After performing feature selection, correlation, welch t-test, OLS regression and logistic
regression in the dataset of Polish companies it can be inferred that when predicting bankruptcy
of companies, single variable cannot justify the result. It is very important to consider several
variables while forecasting the financial condition of a company. Also, each year have different
factors which are significant for predicting the bankruptcy. After doing feature selection, it was
seen that none of the 64 attributes were insignificant to the bankruptcy prediction. Each of the 64
attributes have at least some level of importance. Best five attributes were considered in order to
perform the analysis and some of the most important attributes are ((current assets - inventory) /
short-term liabilities) and (operating expenses/ total liabilities), which also have a positive
relationship in bankrupt companies but no relationship in non-bankrupt companies. When t-test
was done in those 5 attributes, some of attributes showed significance importance for both
bankrupt and non-bankrupt firms. Prediction model built using logistic regression performed
really well on unseen test data specially for year 2 dataset will close to 99% precision. This
model can be used reasonable well to predict bankruptcy in future. Balancing the data before
building the model is very important in this kind of highly unbalanced data.
