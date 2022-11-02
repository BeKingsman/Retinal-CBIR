# Retinal-CBIR

Aim of this project is to detect retinal diseases using CBIR. 
Given an input image, we will query all similar images in our database and based on their diagnosis, we will predict disease in input image.

## Description of all files in this repository

## 1. parse_data.py
This file contains code to parse kaggle dataset and move all images to Images folder in root directory.

## 2. Db.py
This file contains code to interact with mysql database.
First set up a mysql db and update the credentials in db.py.
Then run db.py to create empty tables.

## 3. main.py
This is the main file that contains code for CBIR feature computation and related utility functions.
Run main.py to compute features of all images present in Images folder, computed features will get stored in mysql db.

## 4. gui.py
This file contains code for gui application that is built using python tkinter.
It uses main.py and db.py to query db, compute features, calcutate distance etc.
run gui.py to launch the tkinter application.

## 5. train.py
This file contains code to optimally find the right set of weights for all features, which will be used in main.py.
I used python library to find minima as we need to find weights for which errors get minimised.



## Steps to set up the project
1. Create a virtualenv and install all dependencies.
2. If you are using your own dataset, then update parse_data.py.
3. Create a mysql db on your local machine and update its credentials in db.py.
4. run "python db.py" to create empty tables.
5. run "python main.py" to fill db.
6. run "gui.py" to launch the application

For any queries, feel free to contact me @ harshagarwal8055@gmail.com.

## Some Screenshots of the application
<img width="700" alt="image" src="https://user-images.githubusercontent.com/53928332/199442872-d8f75690-5acd-43dc-b63d-b1ea993e51f3.png">
<img width="700" alt="image" src="https://user-images.githubusercontent.com/53928332/199443219-b2c2be89-eb33-42b5-85e8-6dfc20bdb0a6.png">
<img width="700" alt="image" src="https://user-images.githubusercontent.com/53928332/199443389-a996ce12-dabc-48fa-93b9-326d473c4225.png">
