import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

#X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
#y = np.array([3, 6, 9, 12, 15, 18, 21, 24])
# y = 3x

X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([3, 7, 8, 14, 13, 20, 19, 27])

X_train = X[:6]
y_train = y[:6]

X_test = X[6:]
y_test = y[6:]

model = LinearRegression()

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train MSE:", mean_squared_error(y_train, train_pred))
print("Test MSE:", mean_squared_error(y_test, test_pred))

for degree in [1, 2, 5]:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(
        f"Degree {degree}: "
        f"Train MSE={train_mse:.4f}, "
        f"Test MSE={test_mse:.4f}"
    )
