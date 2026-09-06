import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('StudentMarksDataset.csv',sep=',')
print(f"Head: {df.head()}")

print(f"Shape: {df.shape}")

# Remove rows where Std_StudyHours or Std_Marks is NaN
df = df.dropna(subset=['Std_StudyHours', 'Std_Marks'])

# Check the new shape of the dataframe
print(f"New shape: {df.shape}")

choice = int(input("Enter 0 for full dataset, 1 to filter by branch: "))

if choice == 1:
    # Get unique branches from the Std_Branch column
    branches = df['Std_Branch'].unique()
    print(f"Available branches: {list(branches)}")

    while True:
      # Take the specific branch as input
      target_branch = input("Enter the exact name of the Stream of Study: ")
      if target_branch in branches:
          break
      else:
          print("Invalid branch name. Please try again.")

    # Filter the dataset
    df = df[df['Std_Branch'] == target_branch]
    print(f"Dataset filtered to {target_branch}. New shape: {df.shape}")
else:
    print(f"Using full dataset. Shape: {df.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X=np.array(df['Std_StudyHours']).reshape(-1,1)
Y=np.array(df['Std_Marks']).reshape(-1,1)
print(X.shape,Y.shape)

X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=10)

model=LinearRegression()

model.fit(X_train,Y_train)

print(f"RMSE: {model.score(X_train,Y_train)}")

print(f"RMSE: {model.score(x_test,y_test)}")

y_pred=model.predict(x_test)
print("mean squared error",mean_squared_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))

plt.scatter(df['Std_StudyHours'],Y,color='red')
plt.plot(df['Std_StudyHours'],model.predict(X),color='black')
plt.show()

t1 = int(input("Number of Hours the Student spends studying: "))

m1 = model.predict(np.array([[t1]]))

print(f"Predicted marks for {t1} hours of study: {m1[0][0]:.2f}")

rm1 = int(input("To suggest tips on methods of study improvement, please provide most recent mark: "))

if 0 <= t1 < 6.33:
    if rm1 < 76:
        print("Low hours, Low marks: Try to increase your study consistency and seek help with core concepts.")
    elif 76 <= rm1 < 84:
        print("Low hours, Mid marks: You are efficient, but increasing your hours slightly could boost you to the top tier.")
    elif 84 <= rm1 <= 100:
        print("Low hours, High marks: Exceptional efficiency! Ensure you aren't skipping deep details that might appear in harder exams.")
elif 6.33 <= t1 < 7.67:
    if 0 <= rm1 < 76:
        print("Mid hours, Low marks: Your effort is there, but your study methods might be ineffective. Try active recall.")
    elif 76 <= rm1 < 84:
        print("Mid hours, Mid marks: You are on the right track. Focus on optimizing your schedule to break into the 80s.")
    elif 84 <= rm1 <= 100:
        print("Mid hours, High marks: Great balance. Keep maintaining this pace to stay in the top percentile.")
elif 7.67 <= t1:
    if 0 <= rm1 < 76:
        print("High hours, Low marks: High risk of burnout. You are studying hard but not smart. Re-evaluate your resources.")
    elif 76 <= rm1 < 84:
        print("High hours, Mid marks: You have great dedication. Focus on practice tests to convert those hours into higher scores.")
    elif 84 <= rm1 <= 100:
        print("High hours, High marks: Master level. Your dedication is paying off. Help others to further solidify your knowledge.")
else:
    print("Input values are outside the specified range for analysis.")
