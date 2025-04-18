# A demonstration for algorithmic bias in a simple neural network model
import keras
import numpy as np

class Person:
    def __init__(self,credit_history,income,employed,postal_code):
        self.credit_history = credit_history
        self.employed = employed
        self.income = income * employed
        self.postal_code = postal_code

        #supposed simulate how getting a loan may depend on where you live (high income areas vs rather poor areas)
        self.would_get_loan = self.income * 0.0003 * employed + credit_history * 3 + postal_code * 0.1 > 85
    
def create_people(num_people):
    people = []

    for _ in range(num_people):
        credit_history = np.random.randint(0, 5)
        employed = np.random.randint(0, 2)
        income = np.random.randint(0, 100000)
        postal_code = np.random.randint(0, 1000)
        person = Person(credit_history,income, employed, postal_code)
        people.append(person)
    
    return people

def main():
    np.random.seed(42)
    people = create_people(10000)

    # Select sample
    sample = [p for p in people if np.random.rand() > 0.7] # we take a random subpopulation
    X = np.array([[p.credit_history, p.employed, p.income,p.postal_code] for p in sample])
    y = np.array([p.would_get_loan for p in sample])

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(4,)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=200, batch_size=32)


    # Test the model on example data

    print(f"Model input: [credit_history=2,employed=1,income=50_000,postal_code=200], Should they get the load according to the model?: {model.predict(np.array([[2,1,50000,200]]))[0][0] > 0.5}" )
    print(f"Model input: [credit_history=2,employed=1,income=50_000,postal_code=950], Should they get the load according to the model?: {model.predict(np.array([[2,1,50000,950]]))[0][0] > 0.5}" )



if __name__ == "__main__":
    main()