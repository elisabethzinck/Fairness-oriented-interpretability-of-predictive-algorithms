import os
os.getcwd()

file_path = os.path.join( os.getcwd(), '..', 'test', 'test.txt' )

with open(file_path, 'w') as f:
    f.write('virker det?')

