import click
import os

from wildlifeml.utils.io import load_json, save_as_csv, save_as_json


@click.command()
@click.option(
    '--md_file', '-f', help='Path to MD file.', required=True
)
@click.option(
    '--data_dir', '-p', help='Your personal path to this dir.', required=True
)
def main(data_dir: str):

    meta_data = load_json(os.path.join(data_dir, 'channel_islands_camera_traps.json'))
    station_dict = {x['id']: x['location'] for x in meta_data['images']}
    label_dict = {x['image_id']: x['category_id'] for x in meta_data['annotations']}
    meta_dict = {
        k: {'label': label_dict[k], 'station': station_dict[k]}
        for k in station_dict.keys() & label_dict.keys()
    }
    save_as_csv(
        [(f'{k}.jpg', v['label'], v['station']) for k, v in meta_dict.items()],
        os.path.join(data_dir, 'metadata.csv'),
        header=['orig_name', 'true_class', 'station']
    )
    label_map = {x['id']: x['name'] for x in meta_data['categories']}
    save_as_json(label_map, os.path.join(data_dir, 'label_map_names.json'))


if __name__ == '__main__':
    main()
