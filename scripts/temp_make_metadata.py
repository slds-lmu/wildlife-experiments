import click
import os

from wildlifeml.utils.io import load_json, save_as_csv


@click.command()
@click.option(
    '--md_file', '-f', help='Path to MD file.', required=True
)
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
def main(md_file: str, repo_dir: str):

    meta_data = load_json(md_file)
    station_dict = {x['id']: x['location'] for x in meta_data['images']}
    label_dict = {x['image_id']: x['category_id'] for x in meta_data['annotations']}
    meta_dict = {
        k: {'label': label_dict[k], 'station': station_dict[k]}
        for k in station_dict.keys() & label_dict.keys()
    }
    save_as_csv(
        [(k, v['label'], v['station']) for k, v in meta_dict.items()],
        os.path.join(repo_dir, 'data', 'metadata_channel_islands.csv'),
        header=['orig_name', 'true_class', 'station']
    )


if __name__ == '__main__':
    main()
