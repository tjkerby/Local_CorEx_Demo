# Streamlit MNIST CorEx App

A web application for visualizing MNIST digit groups and running CorEx (Correlation Explanation) analysis with delete node experiments.

## Features

- **📊 Group Visualization**: View average pixel values for all 20 digit groups
- **🔧 Train CorEx**: Train CorEx models on selected groups and hidden layers
- **📈 Delete Node Experiment**: Run the delete node experiment to measure factor importance

## Installation

### 1. Install Dependencies

```bash
pip install -r streamlit_requirements.txt
```

### 2. Run the App

From the project root directory:

```bash
streamlit run streamlit_app.py
```

The app will start at `http://localhost:8501`

## Usage

### Group Visualization Tab
- Click "Generate Group Visualizations" to see average pixel values for each of the 20 groups
- View statistics about group size and dominant digit

### Train CorEx Tab
- Select a group (0-19)
- Select a hidden layer (1, 2, or 3)
- Click "Train CorEx Model" to train a CorEx model on that group
- View the total correlation score and top factors by mutual information

### Delete Node Experiment Tab
- Select the same group and layer that you trained a CorEx model on
- Select which factor to delete
- Adjust the number of nodes to drop
- Click "Run Delete Node Experiment" to see how removing neurons affects classifier accuracy

## How It Works

1. **Data Loading**: The app loads pre-trained models and MNIST data on startup
2. **Group Partitioning**: MNIST test set is partitioned into 20 groups using clustering
3. **CorEx Training**: For each group, CorEx learns interpretable factors that explain the relationship between hidden layer outputs and classifier predictions
4. **Delete Node Experiment**: By progressively removing neurons from important factors, the app measures their contribution to classification accuracy

## Performance Notes

- First run will take several minutes to download MNIST and compute initial clusters
- Subsequent runs will be much faster thanks to caching
- Each CorEx training takes 30-60 seconds

## Project Structure

```
Local_CorEx_Demo/
├── streamlit_app.py           # Main Streamlit app
├── streamlit_requirements.txt  # Python dependencies
├── paper_mnist/
│   ├── delete_node_experiment.ipynb  # Original Jupyter notebook
│   ├── nn_utils.py
│   ├── nn_plotting.py
│   ├── mnist_classifier/
│   │   ├── model.py
│   │   ├── data.py
│   │   └── model/
│   │       ├── config.py
│   │       ├── *.ckpt (model checkpoints)
│   │       └── ...
│   └── ...
└── ...
```

## Troubleshooting

### "Dataset not found" error
- The app will automatically download MNIST on first run
- Make sure you have internet connectivity
- Check that the `MNIST` folder has write permissions

### "Module not found" error  
- Make sure you're running the app from the project root directory
- Verify all dependencies are installed with `pip install -r streamlit_requirements.txt`

### Slow performance
- First run: Normal, takes several minutes
- If subsequent runs are slow, try clearing the Streamlit cache:
  ```bash
  streamlit cache clear
  ```

## Key Parameters in Delete Node Experiment

- **Group**: Which digit group to test
- **Hidden Layer**: Which layer's neurons to delete (1=500 units, 2=400 units, 3=300 units)
- **Factor**: Which CorEx factor to target
- **Number of Nodes to Drop**: How many neurons to remove per iteration
