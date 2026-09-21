from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from .artifacts import digest

# D15: resource-exceeding inputs are rejected, never silently downscaled. Below Pillow's
# decompression-bomb warning threshold so the typed rejection fires first.
MAX_SOURCE_PIXELS = 80_000_000


def normalize_image(path):
    raw=Path(path).read_bytes()
    try: source=Image.open(path)
    except UnidentifiedImageError as exc: raise ValueError(f'malformed_image: {exc}') from exc
    except Image.DecompressionBombError as exc: raise ValueError(f'excessive_image_size: {exc}') from exc
    with source:
        if source.format not in ('JPEG','PNG'): raise ValueError('unsupported_image: use JPEG or PNG')
        # D15: no hidden frame selection; APNG and other multi-frame containers are rejected.
        if getattr(source,'n_frames',1)!=1: raise ValueError('multi_frame_image: single-frame JPEG or PNG required')
        w,h=source.size
        if w*h>MAX_SOURCE_PIXELS: raise ValueError(f'excessive_image_size: {w}x{h} exceeds {MAX_SOURCE_PIXELS} pixels')
        orientation=source.getexif().get(274,1)
        if orientation not in range(1,9): raise ValueError('invalid_orientation')
        matrices={1:[[1,0,0],[0,1,0],[0,0,1]],2:[[-1,0,w],[0,1,0],[0,0,1]],3:[[-1,0,w],[0,-1,h],[0,0,1]],4:[[1,0,0],[0,-1,h],[0,0,1]],5:[[0,1,0],[1,0,0],[0,0,1]],6:[[0,-1,h],[1,0,0],[0,0,1]],7:[[0,-1,h],[-1,0,w],[0,0,1]],8:[[0,1,0],[-1,0,w],[0,0,1]]}
        try: oriented=ImageOps.exif_transpose(source)
        except OSError as exc: raise ValueError(f'malformed_image: {exc}') from exc
        has_alpha='A' in oriented.getbands() or 'transparency' in oriented.info
        if has_alpha:
            rgba=oriented.convert('RGBA'); bg=Image.new('RGBA',rgba.size,'white'); bg.alpha_composite(rgba); image=bg.convert('RGB')
        else: image=oriented.convert('RGB')
    metadata={'source_sha256':digest(raw),'pixel_sha256':digest(image.tobytes()),'original_size':[w,h],'normalized_size':list(image.size),'orientation':orientation,'orientation_transform':matrices[orientation],'color_conversion':'RGB','alpha_conversion':'white_composite' if has_alpha else 'none'}
    return image,metadata
