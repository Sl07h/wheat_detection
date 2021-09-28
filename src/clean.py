import os
import shutil

fields = os.listdir('data')
for field in fields:
    months = filter(os.path.isdir, os.listdir('data/'+field))
    print(months)
    for month in months:
        shutil.rmtree('data/{}/{}/mod'.format(field, month), ignore_errors=True)
        shutil.rmtree('data/{}/{}/tmp'.format(field, month), ignore_errors=True)
        try:
            os.remove('data/{}/{}/log/metadata.csv'.format(field, month))
        except:
            print('can\'t remove metadata.csv')