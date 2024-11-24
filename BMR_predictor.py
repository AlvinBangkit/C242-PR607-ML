import tensorflow as tf
import numpy as np
import pandas as pd

CAL_DATA_PATH = "calories.csv"

temp_list = []

cal_set = pd.read_csv(CAL_DATA_PATH)

model = tf.keras.models.load_model("model/BMR_predictor.keras")

activity_factors = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9
}

loop_key = True


def predict_bmr_and_calories(age, weight, height, gender, activity_level):
    # Preprocess user input
    gender_male = int((1 if gender == 'male' else 0))
    gender_female = int((1 if gender == 'female' else 0))

    # Normalize the input values
    """scaler = StandardScaler()
    normalized_values = scaler.transform([[age, weight, height]])
    age_norm, weight_norm, height_norm = normalized_values[0]"""

    # Prepare the input data for prediction
    """input_data = np.array(
        [[age_norm, weight_norm, height_norm, gender_female, gender_male]])"""

    # input_data = np.array([[age, weight, height, gender_female, gender_male]])

    # Predict BMR using the model
    predicted_bmr = model.predict(
        np.array([[age, weight, height, gender_female, gender_male]]), verbose=0).item()

    # Calculate daily calorie needs based on activity level
    calories_needed = predicted_bmr * activity_factors[activity_level]

    # Display the results
    return predicted_bmr, calories_needed


def activity_list(activity):
    print("please type in one of the following :\n")
    n = 1
    for act in activity:
        print(f"{n}.{act}")
        n += 1
    print("\nactivity type :")


def table_bmr():
    age = int(input("Input age :"))
    weight = int(input("Input weight :"))
    height = int(input("Input height :"))
    gender = input("Input gender :")
    activity = input(activity_list(activity_factors))
    BMR, cal = predict_bmr_and_calories(age, weight, height, gender, activity)

    table = [
        f"\n|   {activity}    |".upper(),
        f"> Age    : {age} years  ",
        f"> Weight : {weight} kg",
        f"> Height : {height} cm",
        f"> Gender : {gender}",
        f"> BMR    : {BMR:.2f}   ",
        f"> cal    : {cal:.2f} cal\n"
    ]

    for a in table:
        print(a)


def normalize_food_name(food_name):
    # Mengonversi huruf pertama dari setiap kata menjadi kapital
    return ' '.join([word.capitalize() for word in food_name.split()])

# Fungsi utama untuk menghitung porsi makanan berdasarkan input user


def calculate_food_portion(food_list, total_calories_needed):
    # Validasi jumlah jenis makanan
    if len(food_list) > 10:
        return "Error, maksimal 10 jenis makanan"

    # Normalisasi nama makanan sesuai dengan format di dataset
    normalized_food_list = [normalize_food_name(food) for food in food_list]

    # Filter dataset berdasarkan makanan yang diinput user
    selected_foods = cal_set[cal_set['FoodItem'].isin(normalized_food_list)]

    # Jika ada makanan yang tidak ditemukan di dataset, tampilkan pesan error
    if selected_foods.empty or len(selected_foods) < len(normalized_food_list):
        return "Beberapa makanan yang diinput tidak ditemukan di dataset."

    # Konversi kolom 'Cals_per100grams' ke tipe numerik (menghapus satuan " cal")
    selected_foods['Cals_per100grams'] = selected_foods['Cals_per100grams'].str.replace(
        ' cal', '').astype(float)

    # Membagi kalori secara merata ke setiap jenis makanan yang dipilih
    calories_per_food = total_calories_needed / len(normalized_food_list)

    # Kalkulasi porsi makanan berdasarkan kalori per 100 gram
    portions = {}
    for index, row in selected_foods.iterrows():
        food_name = row['FoodItem']
        calories_per_100g = row['Cals_per100grams']

        # Menghitung gram yang dibutuhkan berdasarkan kalori yang diinginkan
        grams_needed = (calories_per_food / calories_per_100g) * 100
        # Membulatkan hingga 2 desimal
        portions[food_name] = round(grams_needed, 2)

    return portions


def list_maker():
    temp_list = []
    sub_loop = True
    n = 0
    total_calories_needed = int(input("Insert the overall needed calories :"))
    result = 0
    while sub_loop:
        food = input('\ninsert food (type "done" to end):'.lower())
        if food == 'done' or len(temp_list) == 10:
            result = calculate_food_portion(temp_list, total_calories_needed)
            print(f"\n{result}")
            break

        listed = food
        temp_list.append(listed)

        for a in temp_list:
            print(f"{n+1}.{a}")
            n = n+1
        n = 0


def cal_count():
    print("1. Count food\n2. List of food")
    key = input('Input :')

    if key == '1':
        list_maker()
    elif key == '2':
        print(cal_set)
    else:
        print("\nInvalid input!\n")


def menu_choose():
    print("\n1. BMR & Calories Predictor\n2. Calory Counter")

    key = input("Type 'exit' to end session :")

    if key == 'exit':
        loop_key = False
        return loop_key
    elif key == '1':
        table_bmr()
    elif key == '2':
        cal_count()
    else:
        print("\nInvalid input!\n")

    key = None


while loop_key:

    menu_choose()
