import csv
import os

with open('credits.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter= ' ', quotechar = '|')
    for row in reader:
        print(', '.join(row))