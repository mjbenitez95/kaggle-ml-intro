import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

def get_percent_error(value, mean):
  return round(((value/mean) * 100), 2)

def main():
  # Load data
  vehicle_file_path = '../input/vehicle-data/vehicles.csv'
  vehicle_data = pd.read_csv(vehicle_file_path) 

  # Filter rows with missing price values
  filtered_vehicle_data = vehicle_data.dropna(axis=0)
  mean_price = filtered_vehicle_data["MSRP"].mean()
  
  # Choose target and features
  y = filtered_vehicle_data.MSRP
 
  vehicle_features = [
    'Year', 
    'Engine HP', 
    'Engine Cylinders',
    'Highway MPG',
    'City MPG',
    'Popularity',
  ]

  X = filtered_vehicle_data[vehicle_features]

  # Split data into training and validation data
  training_X, validation_X, training_y, validation_y = train_test_split(X, y, random_state = 0)

  # fit and predict using forest model
  vehicle_forest_model = RandomForestRegressor(random_state = 1)
  vehicle_forest_model.fit(training_X, training_y)
  vehicle_forest_predictions = vehicle_forest_model.predict(validation_X)

  # calculate forest model mean absolute error
  forest_mae = mean_absolute_error(validation_y, vehicle_forest_predictions)
  forest_percent_error = get_percent_error(forest_mae, mean_price)
  forest_output = "Random Forest Regression model has a mean absolute average of " + str(forest_mae) + " for a dataset with an average MSRP of " + str(mean_price) + ". This is about " + str(forest_percent_error) + "%."

  print(forest_output)

  # instantiate dict to hold MAEs by number of nodes
  mean_absolute_errors_by_nodes = {}
  
  # populate dict
  for max_nodes in range(10, 1000, 10):
    mae = get_mae(max_nodes, training_X, validation_X, training_y, validation_y)
    mean_absolute_errors_by_nodes[max_nodes] = mae

  # plot data
  plt.plot(mean_absolute_errors_by_nodes.keys(), mean_absolute_errors_by_nodes.values())
  plt.show()

  # find best fit for Tree Regression model
  min_key, min_value = get_mins_in_map(mean_absolute_errors_by_nodes)
  tree_percent_error = get_percent_error(min_value, mean_price)
  tree_output = "Tree Regression model is best fit at " + str(min_key) + " nodes, with a mean absolute error of " + str(min_value) + ". This is about " + str(tree_percent_error) + "%."
  print(tree_output)

if __name__ == "__main__":
  main()
