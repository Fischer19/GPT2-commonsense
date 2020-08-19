# GPT2-commonsense


## Requirements

To install requirements (make sure conda is installed)

```setup
bash dependencies.bash
```

## ConceptNet Experiments:

### Training:

To use the default setting:
```
bash train_script.py
```
The model parameters will be saved in `./language_modeling/model_conceptnet/pytorch_model.bin`

To use customized training setting, please run:
```
python run_language_modeling.py --output_dir=<model_name> --model_type=gpt2 --model_name_or_path=<"gpt2" or path_to_pretrained_model> --do_train --train_data_file=train100k_processed.txt --do_eval --eval_data_file=test_processed.txt --line_by_line --learning_rate 1e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 5000 --evaluate_during_training
```

### Generation:

To use the defaut generation setting:
```
bash generation_script.bash
```
The generation results will be saved in `./language_modeling/results/test_model_conceptnet.txt`

To use customized generation setting, please run:
```
python generation_script.py --model_type gpt2 --model_name_or_path <path_to_saved_model> --length 100 --stop_token "<EOS>" --k 1 --num_return_sequences 1 --test_dir test_prompt.txt --output_dir results/<name_of_generation_file>
```

## Evaluation

### Results Comparing to COMeT


| Method \ Metrics | PPL | AVG Score | N/T sro | N/T o |
|--------| -------- | -------- | -------- | ------- |
|COMeT   | 4.32     | 95.25    | 59.25    | 3.75 |
|GPT-2   | **1.83**     | 72.87    | 53.90    | **8.18**|
|GPT-2-pretrain| 4.40|73.64|**88.75**|**16.6**|
