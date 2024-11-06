from utils.utils import seed_everything, device_check, normalize_label, unnormalize_label

seed_everything(1)
action_label = 30
action = normalize_label(action_label)
print('the normalized a:', action)
print('the unnormalized a:', unnormalize_label(action))