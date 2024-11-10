from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def tune_knn(X, y):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 31))}
    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print(f'best k for knn: {grid_search.best_params_["n_neighbors"]}')
    print(f'best accuracy for knn: {grid_search.best_score_ * 100:.2f}%')
    return grid_search.best_params_['n_neighbors']

def tune_svm(X, y):
    svm = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print(f'best params for svm: {grid_search.best_params_}')
    print(f'best accuracy for svm: {grid_search.best_score_ * 100:.2f}%')
    return grid_search.best_params_
