import csv
with open('train.csv', 'rt') as inp, open('train_edit.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[0] != "vid_frame (540).jpg":
            writer.writerow(row)