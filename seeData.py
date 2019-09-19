import pickle

f=open("./saved_data/training.pkl","rb")
train=pickle.load(f)
fi=open("./saved_data/train.txt","wt")
fis=open("./saved_data/train_label.txt","wt")
dat=train["train_patch"]
lab=train["train_labels"]
# print(train)
print(lab.shape)
for patch in lab:
	for item in patch:
		print(item,file=fis,end=" ")
	print(file=fis)
# for pic in dat:
# 	channel=pic[0]
# 	for row in channel:
# 		for y in row:
# 			print("%.5f"%y,end=" ",file=fi)
# 		print(file=fi)
# 	print(file=fi)
#