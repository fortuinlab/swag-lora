from datasets import load_dataset
import os

# GLUE and SuperGLUE tasks
GLUE_and_SuperGLUE_tasks = [
    "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli",
    "boolq", "wic"
]

#test purposes only
def testing_dataset_download(save_dir='/p/scratch/hai_baylora/data'):
    """
    Simulates the start of dataset downloads for specified tasks.
    
    Args:
    - save_dir: The directory to save the datasets (not actually used in simulation).
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for task in GLUE_and_SuperGLUE_tasks :
        try:
            # Determine the dataset name
            if task in GLUE_and_SuperGLUE_tasks:
                dataset_name = "super_glue" if task in ["boolq", "wic"] else "glue"
            else:
                dataset_name = task  # Use the task name as the dataset name for MCQA tasks

            # Simulate the start of the dataset download
            print(f"Starting download for task '{task}' from dataset '{dataset_name}'...")
            load_dataset(dataset_name, task, streaming=True)

            print(f"Download simulation started for task '{task}'. Moving to the next task...")
        except Exception as e:
            print(f"Error in downloading task '{task}': {e}")


def download_glue_superglue(save_dir='/p/scratch/hai_baylora/data'):
    """
    Downloads datasets for specified tasks and saves them in a local directory.
    
    Args:
    - save_dir: The directory to save the datasets.
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for task in GLUE_and_SuperGLUE_tasks:
        try:
            # Determine the dataset name
            dataset_name = "super_glue" if task in ["boolq", "wic"] else "glue"

            # Download the dataset
            print(f"Downloading dataset for task '{task}' from '{dataset_name}'...")

            # Load the dataset from Hugging Face
            dataset = load_dataset(dataset_name, task)

            # Define the path to save this dataset
            task_save_path = os.path.join(save_dir, f'data_{task}')

            # Save the dataset
            dataset.save_to_disk(task_save_path)

        except Exception as e:
            print(f"Error in downloading task '{task}': {e}")
    
    for task in GLUE_and_SuperGLUE_tasks:
        try:
            # Determine the dataset name
            dataset_name = "super_glue" if task in ["boolq", "wic"] else "glue"

            # Download the dataset
            print(f"Downloading dataset for task '{task}' from '{dataset_name}'...")

            # Load the dataset from Hugging Face
            dataset = load_dataset(dataset_name, task)

            # Define the path to save this dataset
            task_save_path = os.path.join(save_dir, f'data_{task}')

            # Save the dataset
            dataset.save_to_disk(task_save_path)

        except Exception as e:
            print(f"Error in downloading task '{task}': {e}")


def download_mcqa(save_dir='/p/scratch/hai_baylora/data'):
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    dataset = load_dataset("swag", "regular")
    task_save_path = os.path.join(save_dir, f'data_swag')
    dataset.save_to_disk(task_save_path)
  
    dataset = load_dataset('commonsense_qa')
    task_save_path = os.path.join(save_dir, f'data_cqa')
    dataset.save_to_disk(task_save_path)
    
    dataset = load_dataset('openbookqa')
    task_save_path = os.path.join(save_dir, f'data_obqa')
    dataset.save_to_disk(task_save_path)


    dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
    task_save_path = os.path.join(save_dir, f'data_arc-c')
    dataset.save_to_disk(task_save_path)

    dataset = load_dataset('allenai/ai2_arc', 'ARC-Easy')
    task_save_path = os.path.join(save_dir, f'data_arc-e')
    dataset.save_to_disk(task_save_path)
    
    mmlu_subtasks_list = ['abstract_algebra', 'anatomy', 
                            'astronomy', 'business_ethics', 'clinical_knowledge', 
                            'college_biology', 'college_chemistry', 'college_computer_science', 
                            'college_mathematics', 'college_medicine', 'college_physics', 
                            'computer_security', 'conceptual_physics', 'econometrics', 
                            'electrical_engineering', 'elementary_mathematics', 
                            'formal_logic', 'global_facts', 'high_school_biology', 
                            'high_school_chemistry', 'high_school_computer_science',
                            'high_school_european_history', 'high_school_geography',
                            'high_school_government_and_politics', 'high_school_macroeconomics',
                            'high_school_mathematics', 'high_school_microeconomics', 
                            'high_school_physics', 'high_school_psychology', 
                            'high_school_statistics', 'high_school_us_history',
                            'high_school_world_history', 'human_aging', 'human_sexuality',
                            'international_law', 'jurisprudence', 'logical_fallacies', 
                            'machine_learning', 'management', 'marketing', 'medical_genetics', 
                            'miscellaneous', 'moral_disputes', 'moral_scenarios', 
                            'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
                            'professional_law', 'professional_medicine',
                            'professional_psychology', 'public_relations',
                            'security_studies', 'sociology', 'us_foreign_policy',
                            'virology', 'world_religions']
    
    for subtask in mmlu_subtasks_list:
        dataset = load_dataset('cais/mmlu', subtask)
        task_save_path = os.path.join(save_dir, f'data_mmlu_{subtask}')
        dataset.save_to_disk(task_save_path)
  

if __name__ == "__main__":
  # testing_dataset_download()
    download_glue_superglue()
    download_mcqa()