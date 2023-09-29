file = 'adult.test'
newFile = 'updateAdult.test'

with open(file, 'r') as file:
    lines = file.readlines()

lines = lines[1:]
lines = [line.replace('.', '') for line in lines]

with open(newFile, 'w') as file:
    file.writelines(lines)
