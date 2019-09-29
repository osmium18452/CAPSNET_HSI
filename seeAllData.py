from HSI_Data_Preparation import All_data,band
from utils import patch_size
import numpy as np

All_data['patch'] = np.transpose(All_data['patch'], (0, 2, 3, 1))
np.reshape(np.asarray(All_data['patch']), (-1, patch_size*patch_size*band))

for items in All_data["patch"]:
	for i in items:
		print("%.6f"%i,end=" ")
	print()
