from pathlib import Path
from typing import Union


def gen_md_from_configs(config_root_dir: Union[Path, str],
                        target_md_path: Union[Path, str] = 'model_zoo.md',
                        prefix: str = '') -> None:

    def to_path(p: Union[Path, str]) -> Path:
        if isinstance(p, Path):
            return p
        if isinstance(p, str):
            return Path(p)
        raise ValueError(f'Unsupported type: {type(p)}')

    config_root_dir = to_path(config_root_dir)
    target_md_path = to_path(target_md_path)

    readme_path_list = []
    for readme_path in config_root_dir.rglob('README.md'):
        if readme_path.exists():
            config_name = readme_path.parent.name
            path = prefix / readme_path
            readme_path_list.append((config_name, path.as_posix()))

    with target_md_path.open('w', encoding='utf8') as f:
        f.write('# Model Zoo\n\n')
        f.write('## Baselines\n\n')

        for name, path in readme_path_list:
            f.write(f'### {name.upper()}\n\n')
            f.write(
                f'Please refer to [{name.upper()}]({path}) for details.\n\n')


if __name__ == '__main__':
    gen_md_from_configs('configs', 'docs/en/model_zoo.md', '/')
