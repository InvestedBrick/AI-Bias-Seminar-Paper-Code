import keras
import numpy as np

p_ref = 500

class Person:
    def __init__(self, credit_history, income, employed, postal_code):
        self.credit_history = credit_history
        self.employed = employed
        self.income = income * employed
        self.postal_code = postal_code

        # original
        self.y_orig = income*0.0003 + credit_history*3 + postal_code*0.1 > 70

        # counterfactual (fix postal_code = p_ref)
        self.y_cf   = income*0.0003 + credit_history*3 + p_ref*0.1 > 70


def create_people(num_people):
    people = []
    for _ in range(num_people):
        credit_history = np.random.randint(0, 5)
        employed = np.random.randint(0, 2)
        income = np.random.randint(0, 100000)
        postal_code = np.random.randint(0, 1000)
        person = Person(credit_history, income, employed, postal_code)
        people.append(person)
    return people

def main():
    np.random.seed(42)
    people = create_people(10000)

    sample = [p for p in people if np.random.rand() > 0.7]
    
    X = np.array([[p.credit_history, p.employed, p.income] for p in sample])
    y = np.array([p.y_cf for p in sample])

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(3,)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=200, batch_size=32)

    # Test counterfactual fairness

    print(f"Model input: [credit_history=2,employed=1,income=50_000], Should they get the load according to the model?: {model.predict(np.array([[2,1,50000]]))[0][0] > 0.5}" )


if __name__ == "__main__":
    main()
