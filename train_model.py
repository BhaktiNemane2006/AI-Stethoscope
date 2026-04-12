from ml_model import train_model

X = [
    [1,2,3,4],
    [2,3,4,5],
    [5,6,7,8],
    [6,7,8,9]
]

y = [0,0,1,1]

train_model(X, y)

print("✅ Model created!")