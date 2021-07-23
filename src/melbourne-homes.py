import pandas as pd

def main():
  # read in data
  melbourne_file_path = "../input/melbourne-housing-snapshot/melb_data.csv"
  melbourne_data = pd.read_csv(melbourne_file_path)
  
  # remove empty values
  melbourne_data = melbourne_data.dropna(axis=0)
  
  # select certain features and save as X by convention
  melbourne_features = ["Rooms", "Bathroom", "Landsize", "Latitude", "Longtitude"]
  X = melbourne_data[melbourne_features]

  # head() displays the top few rows
  print(X.describe(), "\n\n", X.head())

if __name__ == "__main__":
  main()
