python run_language_modeling.py --output_dir=model_conceptnet --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=train100k_processed.txt --line_by_line --learning_rate 5e-4 --num_train_epochs=10 --overwrite_output_dir --save_steps 5000 --save_total_limit 5
