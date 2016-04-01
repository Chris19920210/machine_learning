import numpy as np
import pickle

# for 2_gram
data_2gram = []
with open("/media/chris/Elements/train_2gramfrequency.csv", 'r') as f:
    for line in f:
        data_2gram.append(line.strip().split(','))

title = data_2gram[0][1:]

class_map = {title[k]: k for k in range(len(title))}

data = data_2gram[1:]
data = [data[k][1:] for k in range(len(data))]
data = np.asarray(data, dtype='uint32')
max_per_col = data.max(axis=0)
bench = max_per_col > 500
remain_list_2gram = ['2_'+str(i) for i, x in enumerate(bench) if x]
data = data[:,bench]
np.save('/home/chris/Microsoft_malware/2_gram', data)

# for bytes count

data_bytes = []
with open("/media/chris/Elements/train_frequency.csv", 'r') as f:
    for line in f:
        data_bytes.append(line.strip().split(','))
data_bytes2 = data_bytes[1:]
data_bytes2 = [data_bytes2[k][1:] for k in range(len(data_bytes2))]
data_bytes2 = np.asarray(data_bytes2, dtype='uint32')
np.save('/home/chris/Microsoft_malware/data_bytes', data_bytes2)

# for asm count

data_asm = []
with open("/media/chris/Elements/train_frequency2.csv", 'r') as f:
    for line in f:
        data_asm.append(line.strip().split(','))
data_asm2 = data_asm[1:]
data_asm2 = [data_asm2[k][1:] for k in range(len(data_asm2))]
data_asm2 = np.asarray(data_asm2, dtype='uint32')
np.save('/home/chris/Microsoft_malware/data_asm', data_asm2)

# combine the the data
data_all = np.concatenate((data_bytes2, data_asm2, data), axis=1)
bytes_list = ['B_'+ str(k) for k in range(256)]
asm_list = ['A_'+str(k) for k in range(256)]
all_list = bytes_list+asm_list+remain_list_2gram

feature_list = open('/home/chris/Microsoft_malware/feature_list.obj', 'wb')
pickle.dump(all_list, feature_list)
np.save('/home/chris/Microsoft_malware/data_all', data_all)




# label
label = []
with open("/media/chris/Elements/trainLabels.csv", 'r') as f:
    for line in f:
        label.append(line.replace('"', '').strip().split(','))

label = label[1:]

label_map = {label[k][0]: label[k][1] for k in range(len(label))}

y =[data_2gram[k][0] for k in range(len(data_2gram))]
y = y[1:]

y = list(map(lambda x: label_map[x],y))

y = np.array(y)

np.save('/home/chris/Microsoft_malware/label_y', y)








