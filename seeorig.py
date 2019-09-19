import scipy.io as scio
ipg=scio.loadmat("./Data/Indian_pines_gt.mat")
f=open("./seedata/label.txt","wt")
print(ipg["indian_pines_gt"])
for each in ipg["indian_pines_gt"]:
	for e in each:
		print("%3d"%e,file=f,end="")
	print(file=f)