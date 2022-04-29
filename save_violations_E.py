import os
import pandas as pd

data = pd.read_excel (r'/home/ninad/Documents/book8.xlsx')
path  = "/home/ninad/Documents/violations_dashboard"

a = data['BlobURL'].tolist()
type_ = "ril_obstruction"
# print(a)
for i in range(0,len(a)):
    os.chdir(path)
    try:
        os.mkdir(type_)
    except:
        pass
    os.chdir(type_)
    os.system('wget {}'.format(a[i]))
    print(i)


