"""
Function to plot ARC/ConceptARC tasks given json file

Usage:
     python arcvis.py AboveBelow1.json

Author: Soumya Banerjee
"""


import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

def visualize_grid(grid, title="Grid Visualization", str_filename="arcexample.png"):
    """
    Function to plot ARC/ConceptARC task given a grid of number
    :param grid: numpy array
    :param title: title of plot
    :param str_filename: filename in which to save
    :return: Null
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="tab10", origin="upper")
    plt.colorbar()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(str_filename)

def plot_arc_tasks_from_file(str_json_filename):
    """
    Function to plot ARC/ConceptARC tasks given a JSON file
    :param json_file: path to the JSON file containing ARC tasks
    :return: Null
    """
    with open(str_json_filename, 'r') as file:
        data = json.load(file)
    
    # Visualize training examples
    for i, example in enumerate(data.get("train", [])):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        
        visualize_grid(input_grid, title=f"Training Input {i+1}", str_filename="arcplots/"+str_json_filename+"_"+f"arc_train_input{i+1}.png")
        visualize_grid(output_grid, title=f"Training Output {i+1}", str_filename="arcplots/"+str_json_filename+"_"+f"arc_train_output{i+1}.png")

    # Visualize test examples
    for i, example in enumerate(data.get("test", [])):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])

        visualize_grid(grid=input_grid, title=f"Test Input {i+1}", str_filename="arcplots/"+str_json_filename+"_"+f"arc_test_input{i+1}.png")
        visualize_grid(grid=output_grid, title=f"Test Output {i+1}", str_filename="arcplots/"+str_json_filename+"_"+f"arc_test_output{i+1}.png")


# Example usage:
if __name__ == "__main__":

    
    plot_arc_tasks_from_file('AboveBelow1.json')

    # parse command line arguments
    #parser = argparse.ArgumentParser(description="Name of json file")
    #parser.add_argument('json_file', type=str, help="Path to json file")
    #args = parser.parse_args()

    #plot_arc_tasks_from_file(args.json_file)
    
    # TODO: pull in all tasks from ConceptARC
    # TODO: call 4o-mini, o3-mini, DeepSeek R1 on these tasks

