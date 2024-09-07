from tasks.yang.task import generate_trials, rules_dict, generate_trials, get_num_rule
from tasks.yang.hp import get_default_hp
import torch
import sys

def generate_data_batch(task_name, ruleset='all', device = 'cuda', include_task_vector = False, hp_override={}):

    hp: dict = get_default_hp(ruleset, task_name)
    hp.update(hp_override)
    rule_train_now = task_name

    trial = generate_trials(
        rule_train_now, 
        hp,
        'random',
        batch_size=hp['batch_size_train']
    )

    # Turn input and target into [batch, time samples, in/output size]
    input = torch.tensor(trial.x).permute(1, 0, 2).to(device)
    target = torch.tensor(trial.y).permute(1, 0, 2).to(device)
    c_mask = torch.tensor(trial.c_mask).to(device)
    y_loc = torch.tensor(trial.y_loc).to(device)

    num_rules = get_num_rule(ruleset)
    if not include_task_vector:
        input = input[:,:,:-num_rules]

    return input, target, c_mask, y_loc



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    for task_name in rules_dict['all']:

        ruleset = 'all'
        input, target, c_mask, y_loc = generate_data_batch(task_name, device='cpu')

        fig, axes = plt.subplots(2)
        axes[0].imshow(input[0].numpy().T)
        axes[1].imshow(target[0].numpy().T)
        fig.savefig(f'/homes/pr450/repos/puria-RNNs/tasks/yang/example_images/{task_name}.png')

        input_sequence_members, input_counts = torch.unique_consecutive(input, return_counts=True, dim=1)
        input_sequence_members = torch.permute(input_sequence_members, (1, 0, 2))
        input_counts = input_counts.detach().cpu().numpy().tolist()

        target_sequence_members, target_counts = torch.unique_consecutive(target, return_counts=True, dim=1)
        target_sequence_members = torch.permute(target_sequence_members, (1, 0, 2))
        target_counts = target_counts.detach().cpu().numpy().tolist()

        print(task_name, input_counts, target_counts)
