import yaml

with open('configs/part3_bmx_trees_raft.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['input'] = {
    'video_path': None,
    'frames_dir': 'ProPainter/inputs/object_removal/bmx-trees',
    'image_extensions': ['.png', '.jpg', '.jpeg', '.bmp'],
    'max_long_side': 960
}
config['output']['dataset_name'] = 'bmx-trees'
config['segmentation']['prompt_frame_idx'] = 0
config['segmentation']['dynamic_classes'] = ['person', 'bicycle']

with open('configs/part3_bmx_trees_raft.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
