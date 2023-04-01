import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Define global results variable
results = {}


def train_model():
    # Load and preprocess data
    df = pd.read_csv('AAPL.csv')
    df1 = df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split data into training and testing sets
    training_size = int(len(df1)*0.80)
    test_size = len(df1)-training_size
    train_data, test_data = df1[0:training_size,
                                :], df1[training_size:len(df1), :1]

    # Prepare training data for LSTM
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define and compile model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    from tensorflow import keras
    # model = keras.models.load_model(
    #     'C:\\Users\\nithi\\OneDrive\\Desktop\\mlserver\\saved_model.pb')

    # Train model
    model.fit(X_train, y_train, validation_data=(
        X_test, ytest), epochs=20, batch_size=64, verbose=1)
    model.save("C:\\Users\\nithi\\OneDrive\\Desktop\\mlserver\\")

    # Make predictions on training and testing data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Make predictions for next 30 days
    x_input = test_data[152:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 30):
        if (len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)

            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)

            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i+1

    lst_output2 = scaler.inverse_transform(lst_output)

    # Store results in global variable
    global results
    results = {
        'train_predict': train_predict.tolist(),
        'test_predict': test_predict.tolist(),
        'lst_output2': lst_output2.tolist()
    }


# Train the model when the module is loaded
train_model()

# Define run() function to return results


def run():
    global results
    return results
