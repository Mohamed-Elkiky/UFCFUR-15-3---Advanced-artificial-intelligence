## Train the reorder model
python -m task1_purchase_prediction.src.model

## Run quick reorder and prediction demo
python -m task1_purchase_prediction.src.predict

## Evaluate the reorder model (metrics + bias audit)
python -m task1_purchase_prediction.src.evaluate

## Run all test cases
python -m pytest tests/ -v

## Run the Task 1 test suite
python -m pytest tests/test_task1.py -v

## Run the Task 2 test cases
python -m pytest tests/test_task2.py -v

## Run task 2 grading
python -m task2_3_4_cv_quality.src.grading
