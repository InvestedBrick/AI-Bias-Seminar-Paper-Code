# A simple demonstration for sample selection bias in a neural network model
import keras
import numpy as np

AGE_CAP = 45 #if you feel personally offended by this value, feel free to modify it to your liking, however there is no guarantee for the results
class Person:
    #constructs person with age, weight, level of exercise (0 to 1) and diet (0 to 1) where 0 is bad and 1 is good
    def __init__(self,age,weight,excercise,diet):
        self.age = age
        self.weight = weight
        self.excercise = excercise
        self.diet = diet
        self.has_heart_problem = round(age * 0.1 + weight * 0.2 - excercise * 3 - diet * 4 + np.random.normal(-1, 1)) > 25 # the randomness is for genetics

def main():
    people = []
    # Create some people with semi-random attributes
    # with these values ~37.5% of people have heart problems (lets assume it is a data set from the USA :) )
    for _ in range(5000):
        age = np.random.randint(18, 80)
        weight = np.random.randint(50, 150) + age * 0.3
        excercise = np.random.rand()
        diet = excercise + (np.random.rand() - 0.3) * 0.5
        person = Person(age, weight, excercise, diet)
        people.append(person)


    # Select sample

    sample = [p for p in people if p.age < AGE_CAP] #biased sample
    y = np.array([p.has_heart_problem for p in sample])
    X = np.array([[p.age, p.weight, p.excercise, p.diet] for p in sample])

    # Construct as Simple Neural Network
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=200, batch_size=32)

    # Test the model

    test_people = [p for p in people if np.random.rand() > 0.9] # take sample (10%) from the population, including the old people
    model_output = [model.predict(np.array([[p.age, p.weight, p.excercise, p.diet]]))[0][0] > 0.5 for p in test_people]

    n_old = len([p for p in test_people if p.age > AGE_CAP])
    n_young = len(test_people) - n_old

    y_old = 0
    y_young = 0
    
    print("Model output:")
    for i,p in enumerate(test_people):
        if p.has_heart_problem != model_output[i]:
            if p.age > AGE_CAP:
                y_old += 1
            else:
                y_young += 1
    
    acc_old = (1 - y_old / n_old) * 100
    acc_young = (1 - y_young / n_young) * 100
    print(f"Accuracy for old people: {acc_old:.2f}%")
    print(f"Accuracy for young people: {acc_young:.2f}%")

    return (acc_old, acc_young)
            

if __name__ == "__main__":
    total_acc_old = 0.0
    total_acc_young = 0.0
    N_RUNS = 10
    for i in range(N_RUNS): #avarage over N_RUNS runs
        acc_old,acc_young = main()
        total_acc_old += acc_old
        total_acc_young += acc_young   
    
    print(f"Average accuracy for old people: {total_acc_old/N_RUNS :.2f}%")
    print(f"Average accuracy for young people: {total_acc_young/N_RUNS:.2f}%")

    print(f"{((total_acc_young/N_RUNS) - (total_acc_old/N_RUNS)):.2f}% difference")