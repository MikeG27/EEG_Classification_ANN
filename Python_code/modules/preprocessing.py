from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y).transform(y)
    return y


def standarize_data(X_train, X_test):
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)
    return X_train, X_test


def split_data(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)
    return X_train, X_test, y_train, y_test


def process_to_training(subject_data, split):
    X = subject_data.iloc[:, 0:14].values
    y = subject_data.iloc[:, -1].values

    y = encode_labels(y)

    X_train, X_test, y_train, y_test = split_data(X, y, split)
    X_train, X_test = standarize_data(X_train, X_test)

    return X_train, X_test, y_train, y_test
