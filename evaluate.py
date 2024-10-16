import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def generate_classification_report_func(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred)
    print(f'{model_name} Classification Report:\n{report}')

def evaluate_logistic_regression(model, device, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    plot_confusion_matrix(all_labels, all_preds, 'Logistic Regression')
    generate_classification_report_func(all_labels, all_preds, 'Logistic Regression')

def evaluate_models(knn, svm, lr_model, X_test, y_test, device, test_loader):
    print("\nKNN Model Evaluation:")
    knn_preds = knn.predict(X_test)
    plot_confusion_matrix(y_test, knn_preds, 'KNN')
    generate_classification_report_func(y_test, knn_preds, 'KNN')
    
    print("\nSVM Model Evaluation:")
    svm_preds = svm.predict(X_test)
    plot_confusion_matrix(y_test, svm_preds, 'SVM')
    generate_classification_report_func(y_test, svm_preds, 'SVM')
    
    print("\nLogistic Regression Model Evaluation:")
    evaluate_logistic_regression(lr_model, device, test_loader)
