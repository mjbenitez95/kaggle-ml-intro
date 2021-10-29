import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
  model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
  model.fit(train_X, train_y)
  predicted_values = model.predict(val_X)
  return mean_absolute_error(val_y, predicted_values)

def get_mins_in_map(map):
  min_key = min(map, key = map.get)
  return min_key, map[min_key]

def main():
  # Load data
  melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
  melbourne_data = pd.read_csv(melbourne_file_path) 

  # Filter rows with missing price values
  filtered_melbourne_data = melbourne_data.dropna(axis=0)
  
  # Choose target and features
  y = filtered_melbourne_data.Price
  melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                          'YearBuilt', 'Latitude', 'Longtitude']
  X = filtered_melbourne_data[melbourne_features]

  # Split data into training and validation data
  training_X, validation_X, training_y, validation_y = train_test_split(X, y, random_state = 0)

  # instantiate dict to hold MAEs by number of nodes
  mean_absolute_errors_by_nodes = {}
  
  # populate dict
  for max_nodes in range(100, 1000, 10):
    mean_absolute_error = get_mae(max_nodes, training_X, validation_X, training_y, validation_y)
    mean_absolute_errors_by_nodes[max_nodes] = mean_absolute_error

  # find best fit
  min_key, min_value = get_mins_in_map(mean_absolute_errors_by_nodes)
  print("Best fit is at " + str(min_key) + " nodes, with a mean absolute error of " + str(min_value))

if __name__ == "__main__":
  main()
