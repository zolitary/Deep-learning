import os
import shutil
import web

 #筛除多行的label.txt
#
# input_label_dir1 = './data_c/chemistry_formula_images_grey_labels/'
# output_label_dir1 = './data_c/no_multi/'
#
# label_name_list1 = os.listdir(input_label_dir1)
#
# for label_name in label_name_list1:
#     print("label", label_name)
#     label_file_name = input_label_dir1 + label_name
#
#     with open(label_file_name, 'r', encoding='utf-8') as f1:
#         lines = f1.readlines()
#     print(lines)
#     if len(lines) > 1:
#         print(lines[1])
#         shutil.copy(label_file_name, './data/mult-line_label/' + label_name)
#         continue
#     shutil.copy(label_file_name, output_label_dir1 + label_name)
#
# # 筛除多行的label.txt end


# 筛除error mathpix

input_label_dir2 = './data_c/no_multi/'
output_label_dir2 = './data_c/no_error_mathpix/'

label_name_list = os.listdir(input_label_dir2)

for label_name in label_name_list:
    print("label",label_name)
    label_file_name = input_label_dir2 + label_name
    with open(label_file_name, 'r', encoding='utf-8') as f1:
        content = f1.read()
    if 'error mathpix' in content:
        print(content)
        continue
    shutil.copy(label_file_name, output_label_dir2 + label_name)

# 筛除error mathpix end
