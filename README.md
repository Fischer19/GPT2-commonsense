# GPT2-commonsense


## Requirements

To install requirements (make sure conda is installed)

```setup
bash dependencies.sh
```

## Training models on ConceptNet data:

Use the default setting:
```
bash train_script.py
```
The model parameters will be saved in ./model_conceptnet/pytorch_model.bin

To use customized training setting, please run:
```
python run_language_modeling.py --output_dir=<model_name> --model_type=gpt2 --model_name_or_path=<"gpt2" or path_to_pretrained_model> --do_train --train_data_file=train100k_processed.txt --do_eval --eval_data_file=test_processed.txt --line_by_line --learning_rate 1e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 5000 --evaluate_during_training
```

