import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from tensorflow.keras.utils import to_categorical
import re

csv_file = "//content//image_data_with_labels.csv"
df = pd.read_csv(csv_file, header=None)

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values   

X = X.reshape(-1, 64, 64, 3)
y = to_categorical(y, num_classes=5)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

def build_cnn_model():
    model = Sequential([ 
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_model()
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test), verbose=0)

account_balances = {0: 10000, 1: 15000, 2: 20000, 3: 25000, 4: 30000}

def load_user_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img)
    img = img.astype('float32') / 255.0
    return img

def compare_images(user_image, X):
    distances = []
    user_image_flat = user_image.flatten()
    for image in X:
        image_flat = image.flatten()
        dist = euclidean(user_image_flat, image_flat)
        distances.append(dist)
    return np.argmin(distances)

def check_balance(a):
    return a

def deposit(a):
    amount = int(input("Enter the amount to deposit: "))
    a += amount
    print("Your amount has been deposited successfully.")
    print("Your total amount is:", a)
    return a

def withdraw(a):
    amount = int(input("Enter the amount to withdraw: "))
    if (a - amount) >= 0:
        a -= amount
        print("Your amount has been withdrawn successfully.")
        print("Your total amount is:", a)
    else:
        print("Insufficient balance. Your current amount is:", a)
    return a

def parse_command(command, a):
    command = command.lower()
    if re.search(r'check|balance', command):
        print("Your balance is:", check_balance(a))
    elif re.search(r'deposit', command):
        a = deposit(a)
    elif re.search(r'withdraw', command):
        a = withdraw(a)
    elif re.search(r'exit|quit', command):
        print("Thank you for visiting. Have a great day!")
        return False, a
    else:
        print("Invalid command. Please try again.")
    return True, a

def options(n, a):
    print("Welcome,", n)
    while True:
        command = input("Please enter your command (check balance, deposit, withdraw, or exit): ")
        repeat, a = parse_command(command, a)
        if not repeat:
            break
    return a

def atm_system_with_image_comparison():
    while True:  
        print("Please provide an image path for face recognition.")
        image_path = input("Enter the path to your image: ")
        try:
            user_image = load_user_image(image_path)
            predicted_label = compare_images(user_image, X)
            if predicted_label in account_balances:
                print(f"Face recognized! Welcome to the ATM system.")
                a = account_balances[predicted_label]  
                
                hand_position_correct = False
                while not hand_position_correct:
                    print("Please position your hand in front of the display.")
                    hand_position = input("Is your hand in the center, left, or right? ").strip().lower()
                    
                    if hand_position == "center":
                        print("Hand positioned correctly. You can write.")
                        hand_position_correct = True
                    elif hand_position == "left":
                        print("Please move your hand to the right and continue writing.")
                    elif hand_position == "right":
                        print("Please move your hand to the left and continue writing.")
                    else:
                        print("Invalid input. Please enter 'center', 'left', or 'right'.")
                
                a = options(f"User {predicted_label}", a)  
                account_balances[predicted_label] = a
                break
            else:
                print("Face not recognized. Access denied.")
                break
        except Exception as e:
            print(f"Error in processing the image: {e}")
            print("Please make sure the path is correct and try again.")
            continue

if __name__ == "__main__":
    atm_system_with_image_comparison()
