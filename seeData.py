import pickle

f=open("./saved_data/training.pkl","rb")
train=pickle.load(f)
fi=open("./saved_data/train.txt","wt")
dat=train["train_patch"]
lab=train["train_labels"]
# print(train)
print(lab.shape)
# for pic in dat:
# 	channel=pic[0]
# 	for row in channel:
# 		for y in row:
# 			print("%.5f"%y,end=" ",file=fi)
# 		print(file=fi)
# 	print(file=fi)
#