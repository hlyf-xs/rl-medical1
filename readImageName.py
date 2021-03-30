import os
dir1='data/innerLV/image/'#图片文件存放地址
txt1 = 'data/innerLV/image_files.txt'#图片文件名存放txt文件地址
f1 = open(txt1, 'a') #打开文件流

for filename in os.listdir(dir1):
    # print("filename:", filename)


    # filename = filename[0] + filename[len(filename) - 1]
    filename = dir1 + filename
    print(filename)

    f1.write(filename)#只保存名字，去除后缀.jpg
    f1.write("\n")#换行
f1.close()#关闭文件流
