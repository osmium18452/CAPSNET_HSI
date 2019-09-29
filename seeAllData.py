from HSI_Data_Preparation import All_data
import numpy as np

All_data['patch'] = np.transpose(All_data['patch'], (0, 2, 3, 1))
All_data['patch'] = np.reshape(np.asarray(All_data['patch']), (-1, 9 * 9 * 220))
print(All_data["patch"].shape)

f = open("./102data.txt", "w+")
for items in All_data["patch"][8000:8100,:]:
	for i in items:
		print("%.6f" % i, file=f, end=" ")
	print(file=f)
