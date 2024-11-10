import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import load_data, get_numpy_data
from models import LogisticRegression
from hyperparameter_tuning import tune_knn, tune_svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from evaluate import evaluate_models


def train_logistic_regression(device, train_loader, test_loader):
    model = LogisticRegression(28*28, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Logistic Regression Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}')
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Logistic Regression Accuracy: {accuracy:.2f}%')
    return model

def train_knn(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'KNN (k={n_neighbors}) Accuracy: {accuracy * 100:.2f}%')
    return knn

def train_svm_model(X_train, y_train, X_test, y_test, svm_params):
    svm = SVC(kernel=svm_params['kernel'], C=svm_params['C'], gamma=svm_params.get('gamma', 'scale'), probability=True)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"SVM ({svm_params}) Accuracy: {accuracy * 100:.2f}%")
    return svm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, test_loader = load_data(batch_size=64)
    X_train, y_train, X_test, y_test = get_numpy_data()
    
    print("Tuning KNN model hyperparameters...")
    best_k = tune_knn(X_train, y_train)
    
    print("Tuning SVM model hyperparameters...")
    best_svm_params = tune_svm(X_train, y_train)
    
    print("\nTraining KNN model...")
    knn_model = train_knn(X_train, y_train, X_test, y_test, best_k)
    
    print("\nTraining SVM model...")
    svm_model = train_svm_model(X_train, y_train, X_test, y_test, best_svm_params)
    
    print("\nTraining Logistic Regression model...")
    lr_model = train_logistic_regression(device, train_loader, test_loader)
    
    torch.save(lr_model.state_dict(), 'logistic_regression.pth')

    evaluate_models(knn_model, svm_model, lr_model, X_test, y_test, device, test_loader)

if __name__ == '__main__':
    main()
