import HSI_Data_Preparation

Training_data, Test_data = HSI_Data_Preparation.Prepare_data()
f=open("./103data.txt","w+")

print("training data",file=f)
for items in Training_data["train_labels"]:
	print(items,file=f)

print(file=f)
print("_________________________________________",file=f)
print(file=f)
for items in Test_data["test_labels"]:
	print(items,file=f)
