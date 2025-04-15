# A demonstration for algorithmic bias in a simple neural network model
import tensorflow as tf
import keras
import numpy as np

class Person:
    def __init__(self,education_level,employed,postal_code):
        self.education_level = education_level
        self.employed = employed
        self.postal_code = postal_code

        #supposed simulate how getting a loan may depend on where you live
        self.would_get_loan = education_level * 1.5 +  employed * 5 + postal_code/100 * 3 > 25
    
def create_people(num_people):
    people = []

    for _ in range(num_people):
        education_level = np.random.randint(0, 5)
        employed = np.random.randint(0, 2)
        postal_code = np.random.randint(0, 1000)
        person = Person(education_level, employed, postal_code)
        people.append(person)
    
    return people

def main():
    np.random.seed(42)
    people = create_people(10000)

    # Select sample
    sample = [p for p in people if np.random.rand() > 0.4] # we take a random subpopulation
    X = np.array([[p.education_level, p.employed, p.postal_code] for p in sample])
    y = np.array([p.would_get_loan for p in sample])

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(3,)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=50, batch_size=32)


    # Test the model on example data

    print(f"Model input: [education=3,employed=1,postal_code=200], Should they get the load according to the model?: {model.predict(np.array([[3,1,200]]))[0][0] > 0.5}" )
    print(f"Model input: [education=3,employed=1,postal_code=950], Should they get the load according to the model?: {model.predict(np.array([[3,1,950]]))[0][0] > 0.5}" )



if __name__ == "__main__":
    main()