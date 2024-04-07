
# please modify the base path
base_path = "xxx"

protocols = ['p1', 'p2.1', 'p2.2']

w_p1 = open(f'{base_path}/cvpr2024/data/p1/train_dev_label.txt', 'w')
w_p21 = open(f'{base_path}/cvpr2024/data/p2.1/train_dev_label.txt', 'w')
w_p22 = open(f'{base_path}/cvpr2024/data/p2.2/train_dev_label.txt', 'w')


for protocol in protocols:
    if protocol == 'p1':
        train_label = open(f'{base_path}/cvpr2024/data/{protocol}/train_label.txt').readlines()
        dev_label = open(f'{base_path}/cvpr2024/data/{protocol}/dev_label.txt').readlines()
        for line in train_label:
            w_p1.write(line)
        for line in dev_label:
            w_p1.write(line)
    # p2.1 only use dev live data, label == 0
    elif protocol == 'p2.1':
        train_label = open(f'{base_path}/cvpr2024/data/{protocol}/train_label.txt').readlines()
        dev_label = open(f'{base_path}/cvpr2024/data/{protocol}/dev_label.txt').readlines()
        for line in train_label:
            w_p21.write(line)
        for line in dev_label:
            path, label = line.strip().split()
            if int(label) == 0:
                w_p21.write(line)
    # p2.2 only use dev live data, label == 0
    elif protocol == 'p2.2':
        train_label = open(f'{base_path}/cvpr2024/data/{protocol}/train_label.txt').readlines()
        dev_label = open(f'{base_path}/cvpr2024/data/{protocol}/dev_label.txt').readlines()
        for line in train_label:
            w_p22.write(line)
        for line in dev_label:
            path, label = line.strip().split()
            if int(label) == 0:
                w_p22.write(line)






