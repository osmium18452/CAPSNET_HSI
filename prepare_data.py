import HSI_Data_Preparation
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="HSI data preparation")
parser.add_argument("-d", "--directory", default="./saved_data",
					help="Directory the data saved.")
args=parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

Training_data, Test_data = HSI_Data_Preparation.Prepare_data()

output=open(os.path.join(args.directory,"training.pkl"),"wb")
pickle.dump(Training_data,output)

output=open(os.path.join(args.directory,"test.pkl"),"wb")
pickle.dump(Test_data,output)