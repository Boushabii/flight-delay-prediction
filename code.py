import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Flight_delay.csv")
print(df.head())

# Keep only the columns relevant to predicting delays 
df = df[['DayOfWeek','Date','DepTime','Airline','Origin','Dest','CarrierDelay']]
print(df)

# Check for missing values
print(df.isnull().sum())  

# Convert Date from string to datetime so we can extract month and day
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True) 
df['month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df = df.drop(columns = ['Date'])

categories = df.select_dtypes(include = ['object']).columns
print(categories)

# One-hot encode text columns
df_encoded = pd.get_dummies(df, drop_first = True)

# Create the target variable: 1 if delayed 30+ minutes, 0 otherwise
df_encoded['is_delayed_30+'] = np.where(df_encoded['CarrierDelay'] > 30, 1, 0)

X = df_encoded.drop(columns = ['is_delayed_30+', 'CarrierDelay'])
y = df_encoded['is_delayed_30+']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Rebuild a readable version of the training set for analysis
train_set = pd.concat([X_train, y_train], axis = 1)  

print(categories)

for category in categories:
    one_hot_columns = [col for col in train_set.columns if col.startswith(f'{category}_')]
    train_set[category] = train_set[one_hot_columns].idxmax(axis = 1)
    train_set = train_set.drop(columns = one_hot_columns)
    train_set[category] = train_set[category].str.replace(f'{category}_','')

print(train_set)



train_set['is_delayed_30+'].value_counts()

train_set['is_delayed_30+'].mean()

train_set.groupby('Airline')['is_delayed_30+'].mean().sort_values(ascending = False).round(3)*100


DayOfWeek_pct_delayed = train_set.groupby('DayOfWeek')['is_delayed_30+'].mean().round(3)*100
print(DayOfWeek_pct_delayed)


pct_delay_by_origin = train_set.groupby('Origin')['is_delayed_30+'].mean().sort_values(ascending = False).round(3)*100
pct_delay_by_origin.head(20)


# Histogram showing how delay rates are distributed across all airports
plt.figure(figsize=(10, 6))
plt.hist(pct_delay_by_origin.values, bins = 25, color = 'blue', edgecolor = 'black')

plt.title("Distribution of 30+ Minute Delays By Origins", fontsize = 14)
plt.xlabel("Percentage of 30+ Minute Delays (%)", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)

plt.show()



# Train XGBoost with default settings as a baseline
xgb_model = xgb.XGBClassifier(random_state = 0, eval_metric = 'logloss')
print(xgb_model)

xgb_model.fit(X_train, y_train)  

y_pred = xgb_model.predict(X_test)


print("XGBoost Classifier (Baseline):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(cm)


y_pred_proba = xgb_model.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label = f'AUC = {auc_score:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.grid(True)
plt.show()



param_grid = {
    'learning_rate' : [0.01, 0.2],
    'max_depth' : [3, 5, 7],
    'n_estimators' : [100, 250],
    'subsample' : [0.6, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state = 0 , eval_metric = 'logloss')


grid_search = GridSearchCV(estimator = xgb_model,
                           param_grid = param_grid,
                           cv = 3,
                           scoring = 'roc_auc',
                           verbose = 1,
                           n_jobs = -1)



grid_search.fit(X_train, y_train)



print("Best parameters found: ", grid_search.best_params_)

y_pred_best = grid_search.best_estimator_.predict(X_test)

print("XGBoost Classifier (Tuned):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

cm = confusion_matrix(y_test, y_pred_best)
print(cm)


y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label = f'AUC = {auc_score:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.grid(True)
plt.show()