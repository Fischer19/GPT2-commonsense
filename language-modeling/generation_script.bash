python .\run_generation.py --model_type gpt2 --model_name_or_path model_conceptnet --length 100 --stop_token "<EOS>" --k 3 --num_return_sequences 1 --test_dir test_prompt.txt --output_dir results/test_conceptnet_model.txt