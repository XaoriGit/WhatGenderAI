from keras import Sequential
from keras.src.layers import Embedding, LSTM, Dropout, Dense

import FormatingDataset

data = FormatingDataset.get_dataset()

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=data['input_dim'], output_dim=8, input_length=data['max_length']))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(data['x'], data['y'], epochs=10, batch_size=4, validation_split=0.2)

loss, accuracy = model.evaluate(data['x'], data['y'])
print(f'Test Accuracy: {accuracy}, loss: {loss}')

print(model.predict(FormatingDataset.get_array('Дюха')))

model.save('ai.keras')




