"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from werkzeug import secure_filename
from foi_ai import app
import os
import pandas as pd
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Source: healthyeating.com
default_calorie_per_meal = 200

# read input data
def read_data(filename, col):
    word_list = list()
    data_path="/home/nikesh/Desktop/foi_ai/"
    file_path = data_path + filename
    feature_vector = list()
    
    # read the user input file
    fp = open(file_path, mode="r")
    doc = fp.readlines()
    fp.close()
    
    # read english filter words
    fp1 = open("/home/nikesh/Desktop/foi_ai/foi_ai/static/data/stop_words_english.txt", mode="r")
    # get the english stop words to filter the ingredients list
    stop_words = fp1.readlines()
    fp1.close()
    stop_words = [x.strip("\n") for x in stop_words]
    
    # do some action - form a feature vector
    # get all the ingredients
    for x in doc:
        temp = x.strip(" ").split(" ")
        for y in temp:
            y = y.strip(" ")
            y = y.strip(",")
            y = y.strip("\n")
            y = y.upper()
            word_list.append(y)
    
    # get the usefull ingredient list
    for x in word_list:
        if x.lower() in stop_words or len(x) < 4:
            word_list.remove(x)
    
    # select feature dimention
    for x in col:
        if x == "Zone":
            col.remove(x)
    
    # generate the feature vector
    for x in col:
        if x in word_list:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    
    # convert vector into a dataframe
    test_df = pd.DataFrame(feature_vector)
    test_df = test_df.transpose()
    
    return test_df, word_list


# Calorie Computation
def find_calorie_value(ingredients_list):
    cal_df = pd.DataFrame.from_csv("/home/nikesh/Desktop/foi_ai/foi_ai/static/data/ing_calories.csv", header=0)
    # stage 1
    # Form a dictionary of ingredients and their value
    ing_cal_dict = dict()
    for i, row in cal_df.iterrows():
        val = row["Calorie"]
        ing = str(row["Ingredient"])
        ing = ing.upper()
        if ing not in ing_cal_dict:
            ing_cal_dict[ing] = val
        else:
            continue
    # stage 2
    calorie_value = 0
    cnt = 0
    for x in ingredients_list:
        if x in ing_cal_dict:
            calorie_value += ing_cal_dict[x]
            cnt += 1
    calorie_value /= (cnt * 0.8)
    return calorie_value

# side dishes collection
east_side_dishes = [
    "Grapes",
    "Watermelon",
    "Avocado",
    "Litchi",
    "Jamun",
    "Curd",
    "Green Been Salad",
    "Leafy Salad",
    "Green Apple Salad"
]

west_side_dishes = [
    "Banana",
    "Papaya",
    "Chickoo",
    "Guava",
    "Dates",
    "Cashew",
    "Pashion Fruit",
    "Curd",
    "Green Been Salad",
    "Leafy Salad",
    "Green Apple Salad"
]

north_side_dishes = [
    "Cherry",
    "Apple",
    "Apricot",
    "Walnuts",
    "Peach",
    "Plum",
    "Kiwi",
    "Strawberry",
    "Curd",
    "Green Been Salad",
    "Leafy Salad",
    "Green Apple Salad"
]

south_side_dishes = [
    "Jackfruit",
    "Mango",
    "Custard Apple",
    "Sapotta",
    "Coconut",
    "Curd",
    "Green Been Salad",
    "Leafy Salad",
    "Green Apple Salad"
]


def compute_diff_calorie(calorie_value, zone_flag):
    diff_val = default_calorie_per_meal - calorie_value
    if diff_val < 0:
        diff_val *= -1
    
    df = pd.read_csv("/home/nikesh/Desktop/foi_ai/foi_ai/static/data/fruit_calories.csv", header=0)
    df = df.reset_index()
    dish = "Try food suplements like "

    if zone_flag == 1:
        df = df[df["region"] == "East"]
    elif zone_flag == 2:
        df = df[df["region"] == "West"]
    elif zone_flag == 3:
        df = df[df["region"] == "North"]
    elif zone_flag == 4:
        df = df[df["region"] == "South"]

    for i, row in df.iterrows():
        if row["calorie"] > diff_val:
            total_gm = diff_val / row["calorie"]
            total_gm *= 100
            total_gm = round(total_gm, 2)
            dish = dish + str(total_gm) + " grams of "
            dish += row["fruit"]
            dish += " for extra "
            diff_val_round = round(diff_val, 2)
            dish += str(diff_val_round)
            dish += " calories."
            break
    return dish


def tip_generator(calorie_value, zone_flag):
    all_tip = ["Excellent Calorie content. keep up the healthy diet. Include side dishes like ",
    "Good Calorie content. For extra energy try fruits at regular interval. Try something like ",
    "Poor Calorie content. Try energy rich food. Include food items like "]
    tip = ""
    
    if calorie_value - default_calorie_per_meal >= 0:
        tip += all_tip[0]
    elif calorie_value - default_calorie_per_meal < 0 and calorie_value - default_calorie_per_meal > -100:
        tip += all_tip[1]
    else:
        tip += all_tip[2]

    if zone_flag == 1:
        i = 0
        while i < 3:
            randnum = random.randint(0, len(east_side_dishes)-1)
            tip += east_side_dishes[randnum]
            tip += ", "
            i += 1
    elif zone_flag == 2:
        i = 0
        while i < 3:
            randnum = random.randint(0, len(west_side_dishes)-1)
            tip += west_side_dishes[randnum]
            tip += ", "
            i += 1
    elif zone_flag == 3:
        i = 0
        while i < 3:
            randnum = random.randint(0, len(north_side_dishes)-1)
            tip += north_side_dishes[randnum]
            tip += ", "
            i += 1
    else:
        i = 0
        while i < 2:
            randnum = random.randint(0, len(south_side_dishes)-1)
            tip += south_side_dishes[randnum]
            tip += ", "
            i += 1
    tip += "."
    return tip


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.iloc[:, 0:-1]
    Y = balance_data.iloc[:, -1]
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 2107)
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def train_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 1995)
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred


# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root


# Classification Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)


# Driver code
def main(data_frame):
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data_frame)
    clf_gini = train_using_gini(X_train, y_train)
    return clf_gini, X_test, y_test


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year
        )


@app.route("/form")
def form():
    return render_template(
        "form.html"
        )

"""
{"east": 1, "west": 2, "north": 3, "south": 4}
"""

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        usr_name = request.form["username"]
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file.save(secure_filename(file.filename))

        # read data and traing your model
        df = pd.DataFrame.from_csv("/home/nikesh/Desktop/foi_ai/foi_ai/static/data/final-train.csv", header=0)
        col = list(df.columns)
        trained_model, X_test, y_test = main(df)
        y_pred = prediction(X_test, trained_model)

        #acc = accuracy_score(y_test, y_pred)
        # read the user input file and find the feature vector for prediction
        test_df, all_ingredients = read_data(filename, col)
        y_pred_gini = prediction(test_df, trained_model)
        # calorie computation
        calorie_value = find_calorie_value(all_ingredients)
        user_tip = tip_generator(calorie_value, y_pred_gini[0])
        message_str = ""
        if calorie_value < 200:
            message_str = compute_diff_calorie(calorie_value, y_pred_gini[0])

        if y_pred_gini[0] == 1:
            predicted_region = "Eastern Region"
        elif y_pred_gini[0] == 2:
            predicted_region = "Western Region"
        elif y_pred_gini[0] == 3:
            predicted_region = "Northern Region"
        else:
            predicted_region = "Southern Region"
        # render output page
        return render_template(
            "output.html",
            usr=usr_name,
            web_out=predicted_region,
            cal_val=calorie_value,
            tip=user_tip,
            message_str=message_str
            )
