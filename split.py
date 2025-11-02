"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')
    
@cmdline.command()
@click.option('--source',      help='Input ZIP file', metavar='PATH',        type=str, required=True)
@click.option('--dest-prefix', help='Output ZIP files prefix', metavar='PATH', type=str, required=True)
@click.option('--num-splits',  help='Number of splits', metavar='INT',        type=int, required=True)
@click.option('--shuffle', help='Shuffle files before splitting', is_flag=True, default=False)  # 新增shuffle选项

def split(
    source: str,
    dest_prefix: str,
    num_splits: int,
    shuffle: bool,
):
    """Split a ZIP dataset into multiple smaller ZIPs."""
    with zipfile.ZipFile(source, 'r') as src_zip:
        # 获取文件列表
        all_files = [f for f in src_zip.namelist() if f != 'dataset.json']
        total_files = len(all_files)
        
        # 生成索引并打乱（比直接打乱文件列表节省内存）
        indices = np.arange(total_files)
        if shuffle:
            rng = np.random.default_rng(seed=42)
            rng.shuffle(indices)  # 仅打乱索引，内存占用极低

        # 分割索引数组
        chunk_size = (total_files + num_splits - 1) // num_splits
        split_indices = [indices[i*chunk_size : (i+1)*chunk_size] for i in range(num_splits)]

        # 加载标签
        all_labels = {}
        if 'dataset.json' in src_zip.namelist():
            with src_zip.open('dataset.json', 'r') as f:
                metadata = json.load(f)
                if metadata['labels'] is not None:
                    all_labels = {entry[0]: entry[1] for entry in metadata['labels']}

        # 按索引分批处理
        for split_idx in tqdm(range(num_splits), desc='Splitting'):
            current_indices = split_indices[split_idx]
            with zipfile.ZipFile(f"{dest_prefix}_{split_idx:02d}.zip", 'w') as dest_zip:
                # 逐文件处理，避免一次性加载全部文件名
                for idx in tqdm(current_indices, desc=f'Processing split {split_idx}'):
                    file = all_files[idx]  # 按打乱后的索引获取文件名
                    with src_zip.open(file, 'r') as src_file:
                        dest_zip.writestr(file, src_file.read())
                
                # 生成子集标签
                split_files = [all_files[i] for i in current_indices]  # 仅当前分片的文件名
                split_labels = [[file, all_labels[file]] for file in split_files if file in all_labels]
                dest_zip.writestr('dataset.json', json.dumps({'labels': split_labels}))

if __name__ == "__main__":
    cmdline()