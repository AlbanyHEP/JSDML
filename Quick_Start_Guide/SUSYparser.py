#   This parses the SUSY data from the UCI .csv file at (https://archive.ics.uci.edu/ml/datasets/SUSY)
import csv
import numpy as np
sigEvents = []
backEvents = []
with open('SUSY.csv', 'rb') as data:
    rows = csv.reader(data,delimiter=',')
    for row in rows:
        if float(row[0]) == 1.0:
            event = [float(item) for number, item in enumerate(row) if item and (1 <= number <= 18)]
            sigEvents.append(event)
        else:
            event = [float(item) for number, item in enumerate(row) if item and (1 <= number <= 18)]
            backEvents.append(event)

#print(sigEvents)
with open('SUSYSignal.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, delimiter=',')
    for row in sigEvents:
        thedatawriter.writerow(row)

with open('SUSYBackground.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, delimiter=',')
    for row in backEvents:
        thedatawriter.writerow(row)
