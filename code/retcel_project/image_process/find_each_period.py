import dicom
import os
def mode(l):
    # 统计list中各个数值出现的次数
    count_dict = {}
    for i in l:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1
            # 求出现次数的最大值
    max_appear = 0
    for v in count_dict.values():
        if v > max_appear:
            max_appear = v
    if max_appear == 1:
        return
    mode_list = []
    for k, v in count_dict.items():
        if v == max_appear:
            mode_list.append(k)
    print(count_dict)
    return mode_list, count_dict

masker_path = 'D:\PA14_H_6\DCE\JPG_Marker'
maskerfile = os.listdir(masker_path)
maskers_paths = [os.path.join(masker_path, mp) for mp in maskerfile]
maskers_paths = sorted(maskers_paths)

path = 'D:\PA14_H_6\DCE\DCM'
files = os.listdir(path)
file_paths = [os.path.join(path, file) for file in files]
file_paths = sorted(file_paths)

# slice = [dicom.read_file(file_path) for file_path in file_paths]
# slice.sort(key=lambda x: int(x.InstanceNumber))
paths = [pat for pat in zip(file_paths, maskers_paths)]
paths.sort(key=lambda x: int(dicom.read_file(x[0]).InstanceNumber), reverse=False)
print(paths)
slice_lactin = [dicom.read_file(po[0]).SliceLocation for po in paths]
print(slice_lactin)
print(len(slice_lactin))

# loctions = [round(float(slice_lactin[i]) - float(slice_lactin[i+1])) for i in range(len(slice_lactin)-1)]

loctions = []
for i in range(len(slice_lactin)-1):
    temp = round(float(slice_lactin[i]) - float(slice_lactin[i+1]))
    print(temp)
    loctions.append(temp)
print(loctions)
ls = mode(loctions)
print(ls)

pass