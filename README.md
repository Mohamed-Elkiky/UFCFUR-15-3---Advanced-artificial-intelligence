## Train the reorder model
python -m task1_purchase_prediction.src.model

## Run quick reorder and prediction demo
python -m task1_purchase_prediction.src.predict

## Evaluate the reorder model (metrics + bias audit)
python -m task1_purchase_prediction.src.evaluate

## Run the Task 1 test suite
python -m pytest tests/test_task1.py -v

## Run the Task 2 test cases
python -m pytest tests/test_task2.py -v