import pandas as pd

def main():
  # read in data
  iowa_file_path = "../input/home-data-for-ml-course/train.csv"
  home_data = pd.read_csv(iowa_file_path)
  
  print(home_data.describe())

if __name__ == "__main__":
  main()
