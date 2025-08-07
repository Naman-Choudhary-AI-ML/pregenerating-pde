# Training Instructions

## Running the Training Script
To start training, simply run the following command:
```bash
python -m scripts.train.py
```

## Experiment Setup
While running experiments, please ensure that you update the following in the `config.yaml` file:

1. **Dataset Path**: Provide the correct dataset path.
2. **WandB Project Name**: Set the appropriate project name for logging experiments.
3. **Entity Name**: Specify the WandB entity name.
4. **Model Configuration**:
   - **`input_dim`**: Define the input dimension according to your dataset.
   - **`output_dim`**: Set the output dimension correctly.
5. **Model Type**: You have an option to choose between FNO and FFNO

Once these configurations are updated, you are good to go!
