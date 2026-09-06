
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.title("Student Marks Predictor")

df = pd.read_csv("StudentMarksDataset.csv")

st.write("Head:", df.head())
st.write("Shape:", df.shape)

# Remove rows where Std_StudyHours or Std_Marks is NaN.
df = df.dropna(subset=["Std_StudyHours", "Std_Marks"])

st.write("New shape:", df.shape)

choice = st.selectbox(
	"Do you want to use the full dataset or filter by branch?",
	options=["Use full dataset", "Filter by branch"],
)

if choice == "Filter by branch":
	# Remove rows where Std_Branch is NaN.
	df = df.dropna(subset=["Std_Branch"])
	branches = df["Std_Branch"].unique()
	st.write("Available branches:", list(branches))
	branches = [branch for branch in branches if df[df["Std_Branch"] == branch].shape[0] >= 10]
	st.write("Available branches for filteration (restriction on filteration enforced due to lack of datapoints in certain branches, thus only branches with at least 10 data points are included):", list(branches))

	target_branch = st.selectbox(
		"Enter the exact name of the Stream of Study:",
		options=list(branches),
	)

	df = df[df["Std_Branch"] == target_branch]
	st.write(f"Dataset filtered to {target_branch}. New shape:", df.shape)
else:
	st.write("Using full dataset. Shape:", df.shape)

X = np.array(df["Std_StudyHours"]).reshape(-1, 1)
Y = np.array(df["Std_Marks"]).reshape(-1, 1)
st.write("X and Y shapes:", X.shape, Y.shape)

X_train, x_test, Y_train, y_test = train_test_split(
	X,
	Y,
	test_size=0.2,
	shuffle=True,
	random_state=10,
)

model = LinearRegression()
model.fit(X_train, Y_train)

st.write("RMSE:", model.score(X_train, Y_train))
st.write("RMSE:", model.score(x_test, y_test))

y_pred = model.predict(x_test)
st.write("mean squared error", mean_squared_error(y_test, y_pred))
st.write("r2_score", r2_score(y_test, y_pred))

figure, axes = plt.subplots()
axes.scatter(df["Std_StudyHours"], Y, color="red")
axes.plot(df["Std_StudyHours"], model.predict(X), color="black")
st.pyplot(figure)
plt.close(figure)

t1 = st.number_input(
	"Number of Hours the Student spends studying in a week:",
	value=0.00,
	step=0.01
)

m1 = model.predict(np.array([[t1]]))
st.write(f"Predicted marks for {t1} hours of study: {m1[0][0]:.2f}")

rm1 = st.number_input(
	"To suggest tips on methods of study improvement, please provide most recent mark:",
	value=0.00,
	step=0.01
)

if 0 <= t1 < 6.33:
	if rm1 < 76:
		st.write(
			"Low hours, Low marks: Try to increase your study consistency and seek help with core concepts."
		)
	elif 76 <= rm1 < 84:
		st.write(
			"Low hours, Mid marks: You are efficient, but increasing your hours slightly could boost you to the top tier."
		)
	elif 84 <= rm1 <= 100:
		st.write(
			"Low hours, High marks: Exceptional efficiency! Ensure you aren't skipping deep details that might appear in harder exams."
		)
elif 6.33 <= t1 < 7.67:
	if 0 <= rm1 < 76:
		st.write(
			"Mid hours, Low marks: Your effort is there, but your study methods might be ineffective. Try active recall."
		)
	elif 76 <= rm1 < 84:
		st.write(
			"Mid hours, Mid marks: You are on the right track. Focus on optimizing your schedule to break into the 80s."
		)
	elif 84 <= rm1 <= 100:
		st.write(
			"Mid hours, High marks: Great balance. Keep maintaining this pace to stay in the top percentile."
		)
elif 7.67 <= t1:
	if 0 <= rm1 < 76:
		st.write(
			"High hours, Low marks: High risk of burnout. You are studying hard but not smart. Re-evaluate your resources."
		)
	elif 76 <= rm1 < 84:
		st.write(
			"High hours, Mid marks: You have great dedication. Focus on practice tests to convert those hours into higher scores."
		)
	elif 84 <= rm1 <= 100:
		st.write(
			"High hours, High marks: Master level. Your dedication is paying off. Help others to further solidify your knowledge."
		)
else:
	st.write("Input values are outside the specified range for analysis.")
