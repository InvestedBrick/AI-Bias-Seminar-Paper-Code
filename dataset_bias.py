import keras
import numpy as np

class Candidate:
    def __init__(self, gender, education_level, experience_years):
        self.gender = gender  # 0 for female, 1 for male (assuming gender binary to ease calculations)
        self.education_level = education_level  # 0 to 5
        self.experience_years = experience_years  # 0 to 20
        # Simulate hiring bias: males are more likely to be hired
        self.hired = (education_level * 2 + experience_years * 0.5 + gender * 10) > 15

def create_candidates(num_candidates):
    candidates = []
    for _ in range(num_candidates):
        gender = np.random.randint(0, 2)
        education_level = np.random.randint(0, 6)
        experience_years = np.random.randint(0, 21)
        candidate = Candidate(gender, education_level, experience_years)
        candidates.append(candidate)
    return candidates

def main():
    #seed for reproducibility
    np.random.seed(42)
    candidates = create_candidates(10000)

    sample = [c for c in candidates if np.random.rand() > 0.7]
    X = np.array([[c.gender, c.education_level, c.experience_years] for c in sample])
    y = np.array([c.hired for c in sample])

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(3,)),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)

    # Test the model on example data
    print(f"Model input: [gender=female, education=4, experience=10], Hired?: {model.predict(np.array([[0, 4, 10]]))[0][0] > 0.5}")
    print(f"Model input: [gender=male, education=4, experience=10], Hired?: {model.predict(np.array([[1, 4, 10]]))[0][0] > 0.5}")

if __name__ == "__main__":
    main()
