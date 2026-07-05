import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load Dataset
data = pd.read_excel("Dataset for Data Analytics.xlsx")


# Handle Missing Values
data["CouponCode"] = data["CouponCode"].fillna("No Coupon")


# Remove Unnecessary Columns
data = data.drop(columns=[
    "OrderID",
    "CustomerID",
    "TrackingNumber",
    "ShippingAddress",
    "Date"
])


# Split Features and Target
X = data.drop(columns=["OrderStatus"])
y = data["OrderStatus"]


# Encode Categorical Features
encoder = LabelEncoder()

categorical_columns = [
    "Product",
    "PaymentMethod",
    "CouponCode",
    "ReferralSource"
]

for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])


# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create and Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Make Predictions
predictions = model.predict(X_test)


# Evaluate Model
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")