from datasets import load_dataset, concatenate_datasets

def create_custom_dataset(squad_ratio, adversarial_ratio, squad_dataset, adversarial_dataset):
    squad_size = len(squad_dataset)
    num_adversarial = int(squad_size * (adversarial_ratio / (squad_ratio + adversarial_ratio)))

    # Ensure we don't exceed the size of the adversarial dataset
    num_adversarial = min(num_adversarial, len(adversarial_dataset))

    # Select the required number of examples from each dataset
    adversarial_samples = adversarial_dataset.shuffle(seed=42).select(range(num_adversarial))

    # If the number of adversarial samples is less than the expected size, adjust the squad size
    num_squad = squad_size - num_adversarial
    squad_samples = squad_dataset.select(range(num_squad))

    # Concatenate the datasets
    return concatenate_datasets([squad_samples, adversarial_samples])

def align_metadata_schema(example):
    example['metadata'] = None
    return example

# Load datasets
squad = load_dataset('squad', split='train')
adversarial_qa = load_dataset('adversarial_qa', 'adversarialQA', split='train')
squad_eval = load_dataset('squad', split='validation')
adv_qa_eval = load_dataset('adversarial_qa', 'adversarialQA', split='validation')

squad = squad.map(align_metadata_schema)
adversarial_qa = adversarial_qa.map(align_metadata_schema)
squad_eval = squad_eval.map(align_metadata_schema)
adv_qa_eval = adv_qa_eval.map(align_metadata_schema)

# Create datasets with different ratios
dataset_100_0 = squad  # This is just the original SQuAD dataset
dataset_90_10 = create_custom_dataset(90, 10, squad, adversarial_qa)
dataset_75_25 = create_custom_dataset(75, 25, squad, adversarial_qa)
dataset_50_50 = create_custom_dataset(50, 50, squad, adversarial_qa)

validation_100 = squad_eval
validation_50 = create_custom_dataset(50, 50, squad_eval, adv_qa_eval)
validation_0 = adv_qa_eval

# Now you can save these datasets or use them as needed
dataset_100_0.to_json("datasets/train/dataset1000.jsonl")
dataset_90_10.to_json("datasets/train/dataset9010.jsonl")
dataset_75_25.to_json("datasets/train/dataset7525.jsonl")
dataset_50_50.to_json("datasets/train/dataset5050.jsonl")

validation_100.to_json("datasets/validation/validation100.jsonl")
validation_50.to_json("datasets/validation/validation50.jsonl")
validation_0.to_json("datasets/validation/validation0.jsonl")
