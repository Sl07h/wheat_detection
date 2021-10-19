import os
import shutil

fields = os.listdir('data')
for field in fields:
    try:
        months = list(filter(os.path.isdir, os.listdir('data/'+field)))
        print('\n\n')
        print(field)
        print(os.listdir('data/'+field))
        print(months)
        print('data/{}/tmp'.format(field))
        shutil.rmtree('data/{}/mod/'.format(field), ignore_errors=True)
        shutil.rmtree('data/{}/tmp/'.format(field), ignore_errors=True)
        shutil.rmtree('data/{}/veg/'.format(field), ignore_errors=True)
        try:
            os.remove('data/{}/log/metadata.csv'.format(field))
        except:
            print('can\'t remove metadata.csv')
    except:
        print('qwe')